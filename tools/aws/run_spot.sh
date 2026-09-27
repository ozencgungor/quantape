#!/usr/bin/env bash
# Run the Sobol extension on a single AWS instance, end to end:
# bundle -> key pair -> SSH security group -> instance -> build -> run ->
# download result -> terminate.
#
# MARKET=spot (default) or MARKET=on-demand.
#
# Prereqs: aws CLI v2 configured (aws configure), a default VPC in the region.
# Usage:
#   TARGET=300000 ./tools/aws/run_spot.sh
# Env overrides: AWS_REGION, INSTANCE_TYPE, ARCH, TARGET, WINDOW, INPUT_FILE,
#                KEY_NAME, KEY_FILE, LEVEL
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7g.16xlarge}" # 64 vCPU Graviton3 (spot)
ARCH="${ARCH:-arm64}"                          # must match INSTANCE_TYPE: arm64 for c7g/c8g
TARGET="${TARGET:-300000}"
WINDOW="${WINDOW:-128}"
LEVEL="${LEVEL:-1}"
TOOL="${TOOL:-extend_sobol}"
REFINE_FROM="${REFINE_FROM:-21202}"
EPOCHS="${EPOCHS:-2}"
CANDIDATES="${CANDIDATES:-32}"
FRACTION="${FRACTION:-1.0}"
INPUT_FILE="${INPUT_FILE:-internal_docs/new-joe-kuo-7.21201}"
KEY_NAME="${KEY_NAME:-sobol-ext-key}"
KEY_FILE="${KEY_FILE:-$KEY_NAME.pem}"
SHUTDOWN_MIN="${SHUTDOWN_MIN:-300}"
MARKET="${MARKET:-spot}"
BUNDLE="${BUNDLE:-sobol_extender.tar.gz}"
if [ "$TOOL" = "refine_sobol" ]; then
    OUTPUT_NAME="${OUTPUT_NAME:-joe-kuo-refined-${REFINE_FROM}-w${WINDOW}.txt}"
    DONE_MARKER="${DONE_MARKER:-REFINE COMPLETE}"
else
    OUTPUT_NAME="${OUTPUT_NAME:-joe-kuo-extended-${TARGET}-w${WINDOW}.txt}"
    DONE_MARKER="${DONE_MARKER:-Total dimensions}"
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
aws="aws --region $REGION"

# Run log, kept under tools/logs with a descriptive name.
mkdir -p "$repo_root/tools/logs"
LOG_FILE="${LOG_FILE:-$repo_root/tools/logs/sobol-${TOOL}-${MARKET}-d${TARGET}-w${WINDOW}-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "==> log: $LOG_FILE"

echo "==> bundling standalone extender"
"$repo_root/tools/aws/make_bundle.sh" "$repo_root/$BUNDLE" >/dev/null

echo "==> key pair $KEY_NAME"
if ! $aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
    $aws ec2 create-key-pair --key-name "$KEY_NAME" --query KeyMaterial --output text >"$KEY_FILE"
    chmod 600 "$KEY_FILE"
fi

echo "==> SSH security group"
MY_IP="$(curl -s https://checkip.amazonaws.com)"
VPC_ID="$($aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$($aws ec2 describe-security-groups --filters Name=group-name,Values=sobol-ext \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    SG_ID="$($aws ec2 create-security-group --group-name sobol-ext \
        --description "Sobol extender SSH" --vpc-id "$VPC_ID" --query GroupId --output text)"
fi
$aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 \
    --cidr "$MY_IP/32" >/dev/null 2>&1 || true

echo "==> resolving Ubuntu 24.04 AMI ($ARCH)"
AMI="$($aws ssm get-parameter \
    --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/${ARCH}/hvm/ebs-gp3/ami-id" \
    --query Parameter.Value --output text)"

echo "==> launching $MARKET $INSTANCE_TYPE"
if [ "$MARKET" = "on-demand" ]; then
    MARKET_JSON='{"MarketType":"on-demand"}'
else
    MARKET_JSON='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}'
fi
INSTANCE_ID="$($aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options "$MARKET_JSON" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sobol-ext}]' \
    --query 'Instances[0].InstanceId' --output text)"
# Cleanup policy: only terminate after a fully successful run; on any failure
# leave the instance running (self-shutdown bounds cost) so the work can be
# retrieved/resumed. A transient SSH/network blip must never destroy a run.
SUCCESS=0
cleanup() {
    if [ "$SUCCESS" = "1" ]; then
        $aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
    else
        echo "==> WARNING: script did not finish; leaving $INSTANCE_ID running"
        echo "    (self-shutdown after ${SHUTDOWN_MIN} min; terminate manually if not needed)"
    fi
}
trap cleanup EXIT

$aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
IP="$($aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "    instance $INSTANCE_ID at $IP"

echo "==> waiting for SSH"
for _ in $(seq 1 60); do
    if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
        "ubuntu@$IP" true >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

# Safety net: the instance terminates itself after SHUTDOWN_MIN minutes no matter what.
ssh -i "$KEY_FILE" "ubuntu@$IP" "sudo shutdown -h +${SHUTDOWN_MIN}" >/dev/null 2>&1 || true

echo "==> uploading bundle + input"
scp -q -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new \
    "$repo_root/$BUNDLE" "$repo_root/$INPUT_FILE" "ubuntu@$IP:~/"

if [ "$TOOL" = "refine_sobol" ]; then
    REMOTE_ARGS="--input=$(basename "$INPUT_FILE") --output=$OUTPUT_NAME --refine-from=$REFINE_FROM --window=$WINDOW --candidates=$CANDIDATES --epochs=$EPOCHS --fraction=$FRACTION ${TOOL_ARGS:-}"
else
    REMOTE_ARGS="--local --target=$TARGET --level=$LEVEL --window=$WINDOW --input=$(basename "$INPUT_FILE") --output=$OUTPUT_NAME"
fi

echo "==> building and starting $TOOL"
ssh -i "$KEY_FILE" "ubuntu@$IP" '
    set -e
    sudo cloud-init status --wait >/dev/null 2>&1 || true
    if ! command -v g++ >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++
    fi
    tar xzf '"$(basename "$BUNDLE")"' && ./build.sh && \
    ( nohup ./'"$TOOL"' '"$REMOTE_ARGS"' --threads=$(nproc) > run.log 2>&1 </dev/null & )
    echo started' 

echo "==> polling (progress every 30s, partial results downloaded)"
poll_start="$(date +%s)"
while true; do
    sleep 30
    state="$($aws ec2 describe-instances --instance-ids "$INSTANCE_ID"         --query 'Reservations[0].Instances[0].State.Name' --output text)"
    if [ "$state" != "running" ]; then
        echo "---- instance state: ${state} (spot reclaim or shutdown)"
        break
    fi
    if ssh -i "$KEY_FILE" "ubuntu@$IP" "grep -q '$DONE_MARKER' run.log" 2>/dev/null; then
        break
    fi
    elapsed="$(( $(date +%s) - poll_start ))"
    echo "---- elapsed ${elapsed}s"
    ssh -i "$KEY_FILE" "ubuntu@$IP" "tail -n 2 run.log" 2>/dev/null || true
    scp -q -i "$KEY_FILE" "ubuntu@$IP:~/$OUTPUT_NAME" "$repo_root/$OUTPUT_NAME" 2>/dev/null || true
done

echo "==> downloading result"
if scp -q -i "$KEY_FILE" "ubuntu@$IP:~/$OUTPUT_NAME" "$repo_root/$OUTPUT_NAME" 2>/dev/null; then
    echo "==> wrote $repo_root/$OUTPUT_NAME"
    SUCCESS=1
else
    echo "==> no output file (dry run?)"
    SUCCESS=1
fi
ssh -i "$KEY_FILE" "ubuntu@$IP" "tail -n 6 run.log" 2>/dev/null || true
