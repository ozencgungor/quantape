#pragma once

// Schedule: coupon dates generated forward or backward from an effective and
// termination date, with stubs, end-of-month rolls and business-day
// adjustment. ICMA day counts and BUS/252 consume it through DayCounter.

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Frequency.h"
#include "quantape/datetime/Period.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace quantape::datetime {

enum class DateGeneration : std::uint8_t { Forward, Backward, Zero };

class Schedule {
public:
    Schedule() = default;
    Schedule(const Date& effective, const Date& termination, const Period& tenor,
             const Calendar& calendar, BusinessDayConvention convention = BusinessDayConvention::Following,
             DateGeneration rule = DateGeneration::Forward, bool endOfMonth = false)
        : calendar_(calendar), tenor_(tenor), convention_(convention), endOfMonth_(endOfMonth) {
        if (termination < effective) {
            throw std::invalid_argument("Schedule: termination before effective");
        }
        switch (rule) {
            case DateGeneration::Zero:
                unadjusted_ = {effective, termination};
                break;
            case DateGeneration::Forward: {
                unadjusted_.push_back(effective);
                Date next = effective;
                while (true) {
                    next = calendar_.advance(next, tenor_, BusinessDayConvention::Unadjusted,
                                             endOfMonth_);
                    if (next >= termination) {
                        break;
                    }
                    unadjusted_.push_back(next);
                }
                if (unadjusted_.back() != termination) {
                    unadjusted_.push_back(termination);
                }
                break;
            }
            case DateGeneration::Backward: {
                unadjusted_.push_back(termination);
                Date previous = termination;
                while (true) {
                    previous = advanceBackward(previous);
                    if (previous <= effective) {
                        break;
                    }
                    unadjusted_.push_back(previous);
                }
                if (unadjusted_.back() != effective) {
                    unadjusted_.push_back(effective);
                }
                std::reverse(unadjusted_.begin(), unadjusted_.end());
                break;
            }
        }
        adjusted_.reserve(unadjusted_.size());
        for (const Date& date : unadjusted_) {
            adjusted_.push_back(calendar_.adjust(date, convention_));
        }
    }

    const std::vector<Date>& dates() const { return adjusted_; }
    const std::vector<Date>& unadjustedDates() const { return unadjusted_; }
    std::size_t size() const { return adjusted_.size(); }
    const Date& startDate() const { return adjusted_.front(); }
    const Date& endDate() const { return adjusted_.back(); }
    const Period& tenor() const { return tenor_; }
    const Calendar& calendar() const { return calendar_; }
    BusinessDayConvention businessDayConvention() const { return convention_; }
    bool endOfMonth() const { return endOfMonth_; }

    /// True when the period from date(i) to date(i+1) is a full tenor.
    bool isRegular(std::size_t index) const {
        if (index + 1 >= unadjusted_.size()) {
            return false;
        }
        const Date expected = calendar_.advance(unadjusted_[index], tenor_,
                                                BusinessDayConvention::Unadjusted, endOfMonth_);
        return expected == unadjusted_[index + 1];
    }

    /// Coupon frequency implied by the tenor.
    Frequency frequency() const {
        switch (tenorMonths()) {
            case 12: return Frequency::Annual;
            case 6: return Frequency::Semiannual;
            case 3: return Frequency::Quarterly;
            case 2: return Frequency::Bimonthly;
            case 1: return Frequency::Monthly;
            default: return Frequency::Other;
        }
    }

    int tenorMonths() const {
        switch (tenor_.unit()) {
            case TimeUnit::Months: return tenor_.length();
            case TimeUnit::Years: return tenor_.length() * 12;
            default: return 0;
        }
    }

private:
    Date advanceBackward(const Date& date) const {
        switch (tenor_.unit()) {
            case TimeUnit::Days: return date.plusDays(-tenor_.length());
            case TimeUnit::Weeks: return date.plusWeeks(-tenor_.length());
            case TimeUnit::Months: return date.plusMonths(-tenor_.length(), endOfMonth_);
            case TimeUnit::Years: return date.plusYears(-tenor_.length(), endOfMonth_);
        }
        return date;
    }

    Calendar calendar_{};
    Period tenor_{};
    BusinessDayConvention convention_ = BusinessDayConvention::Following;
    bool endOfMonth_ = false;
    std::vector<Date> unadjusted_;
    std::vector<Date> adjusted_;
};

}  // namespace quantape::datetime
