//
// Example usage showing the LAZY EVALUATION pattern with DAY-BASED schedules
// Schedules are specified in days and automatically converted to years for simulation
//

#include <iomanip>
#include <iostream>

#include "CreditGenerator.h"
#include "CreditSimulator.h"
#include "EquityGenerator.h"
#include "EquitySimulator.h"

void exampleEquityLazy() {
    std::cout << "=== Equity Lazy Evaluation Example (Day-Based Schedule) ===" << std::endl;
    std::cout << "Generators trigger simulation automatically on first access!\n" << std::endl;

    // Define time schedule in DAYS (not years!)
    std::vector<int> schedule = {0, 30, 60, 90, 180, 365}; // 0, 1mo, 2mo, 3mo, 6mo, 1yr

    // 1. Create equity simulator
    EquitySimulator simulator(schedule, 42);

    // 2. Add equities with different parameters
    HestonParams aapl(150.0, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7);
    HestonParams msft(300.0, 0.03, 0.06, 1.5, 0.03, 0.25, -0.6);
    HestonParams googl(2800.0, 0.05, 0.04, 2.5, 0.05, 0.35, -0.8);

    simulator.addEquity("AAPL", aapl);
    simulator.addEquity("MSFT", msft);
    simulator.addEquity("GOOGL", googl);

    // 3. Create generator (no simulation yet!)
    EquityGenerator generator(simulator);

    std::cout << "✓ Created simulator and generator" << std::endl;
    std::cout << "✓ No simulation has run yet!\n" << std::endl;

    // 4. Query values - simulation happens automatically!
    std::cout << "Querying AAPL at day 90 (triggers simulation)..." << std::endl;
    double appleSpot = generator.getSpot("AAPL", 90);
    std::cout << "  AAPL spot at day 90: $" << std::fixed << std::setprecision(2) << appleSpot
              << std::endl;

    std::cout << "\nQuerying MSFT at day 365 (triggers simulation)..." << std::endl;
    double msftSpot = generator.getSpot("MSFT", 365);
    double msftVar = generator.getVariance("MSFT", 365);
    std::cout << "  MSFT spot at day 365 (1 year): $" << msftSpot << std::endl;
    std::cout << "  MSFT variance at day 365: " << msftVar << std::endl;

    std::cout << "\nQuerying GOOGL entire path (triggers simulation)..." << std::endl;
    const auto& googlPath = generator.getPath("GOOGL");
    std::cout << "  GOOGL path has " << googlPath.size() << " time points" << std::endl;

    // Display all results
    std::cout << "\n--- All Equity Prices ---" << std::endl;
    for (const auto& name : generator.getEquityNames()) {
        std::cout << "\n" << name << ":" << std::endl;
        for (int day : schedule) {
            std::cout << "  Day " << std::setw(3) << day << ": $" << generator.getSpot(name, day)
                      << std::endl;
        }
    }
}

void exampleCreditLazy() {
    std::cout << "\n=== Credit Lazy Evaluation Example (Day-Based Schedule) ===" << std::endl;

    // Define time schedule in DAYS
    std::vector<int> schedule = {0, 365, 730, 1095, 1460, 1825}; // 0, 1yr, 2yr, 3yr, 4yr, 5yr

    // Create simulator
    CreditSimulator simulator(schedule, 123);

    // Add credits
    CIRParams corpAAA(0.05, 2.0, 0.04, 0.3);
    CIRParams corpBBB(0.08, 1.5, 0.06, 0.4);

    simulator.addCredit("Corp_AAA", corpAAA);
    simulator.addCredit("Corp_BBB", corpBBB);

    // Create generator (no simulation yet!)
    CreditGenerator generator(simulator);

    std::cout << "✓ Created simulator and generator" << std::endl;
    std::cout << "✓ No simulation has run yet!\n" << std::endl;

    // Query values - triggers simulation
    std::cout << "Querying Corp_AAA at day 730 (2 years - triggers simulation)..." << std::endl;
    double aaaRate = generator.getValue("Corp_AAA", 730);
    std::cout << "  Corp_AAA rate: " << std::fixed << std::setprecision(4) << (aaaRate * 100) << "%"
              << std::endl;

    std::cout << "\nQuerying Corp_BBB at day 1095 (3 years - triggers simulation)..." << std::endl;
    double bbbRate = generator.getValue("Corp_BBB", 1095);
    std::cout << "  Corp_BBB rate: " << (bbbRate * 100) << "%" << std::endl;

    // Display all results
    std::cout << "\n--- All Credit Rates ---" << std::endl;
    for (const auto& name : generator.getCreditNames()) {
        std::cout << "\n" << name << ":" << std::endl;
        for (int day : schedule) {
            double rate = generator.getValue(name, day);
            int years = day / 365;
            std::cout << "  Day " << std::setw(4) << day << " (" << years << "yr): " << (rate * 100)
                      << "%" << std::endl;
        }
    }
}

void exampleSelectiveSimulation() {
    std::cout << "\n=== Selective Simulation Example ===" << std::endl;
    std::cout << "Only simulates what you actually query!\n" << std::endl;

    // Schedule: weekly for 3 months
    std::vector<int> schedule = {0, 7, 14, 21, 28, 60, 90};

    // Add many equities
    EquitySimulator simulator(schedule, 999);
    simulator.addEquity("Stock1", HestonParams(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7));
    simulator.addEquity("Stock2", HestonParams(200, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7));
    simulator.addEquity("Stock3", HestonParams(300, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7));
    simulator.addEquity("Stock4", HestonParams(400, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7));

    EquityGenerator generator(simulator);

    // Only query Stock2 and Stock4
    std::cout << "Querying only Stock2 and Stock4 at day 90..." << std::endl;
    std::cout << "Stock2 at day 90: $" << std::fixed << std::setprecision(2)
              << generator.getSpot("Stock2", 90) << std::endl;
    std::cout << "Stock4 at day 90: $" << generator.getSpot("Stock4", 90) << std::endl;

    std::cout << "\n✓ Only Stock2 and Stock4 were simulated!" << std::endl;
    std::cout << "✓ Stock1 and Stock3 were never simulated (not queried)" << std::endl;
}

void exampleExplicitSimulation() {
    std::cout << "\n=== Explicit Simulation Example ===" << std::endl;
    std::cout << "You can still explicitly simulate all if needed\n" << std::endl;

    std::vector<int> schedule = {0, 180, 365}; // 0, 6mo, 1yr

    EquitySimulator simulator(schedule, 555);
    simulator.addEquity("A", HestonParams(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7));
    simulator.addEquity("B", HestonParams(200, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7));

    std::cout << "Calling simulate() explicitly to simulate all equities..." << std::endl;
    simulator.simulate(); // Simulate all equities upfront

    EquityGenerator generator(simulator);

    std::cout << "Now queries don't trigger simulation (already done):" << std::endl;
    std::cout << "A at day 365: $" << std::fixed << std::setprecision(2)
              << generator.getSpot("A", 365) << std::endl;
    std::cout << "B at day 365: $" << generator.getSpot("B", 365) << std::endl;
}

void exampleDayConversion() {
    std::cout << "\n=== Day to Year Conversion Example ===" << std::endl;
    std::cout << "Internally converts days to years using DAYS_IN_YEAR = 365.25\n" << std::endl;

    // Schedule in days
    std::vector<int> schedule = {0, 365, 730}; // 0, ~1yr, ~2yr

    EquitySimulator simulator(schedule, 42);
    HestonParams params(100.0, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7);
    simulator.addEquity("SPX", params);

    EquityGenerator generator(simulator);

    std::cout << "Schedule in days: ";
    for (int day : schedule) {
        std::cout << day << " ";
    }
    std::cout << "\n";

    std::cout << "\nConversion to years:" << std::endl;
    std::cout << "  Day 0 = 0.000 years" << std::endl;
    std::cout << "  Day 365 = " << std::fixed << std::setprecision(3) << (365.0 / 365.25)
              << " years" << std::endl;
    std::cout << "  Day 730 = " << (730.0 / 365.25) << " years" << std::endl;

    std::cout << "\nSimulated values:" << std::endl;
    for (int day : schedule) {
        std::cout << "  Day " << day << ": $" << std::setprecision(2)
                  << generator.getSpot("SPX", day) << std::endl;
    }
}

int main() {
    exampleEquityLazy();
    exampleCreditLazy();
    exampleSelectiveSimulation();
    exampleExplicitSimulation();
    exampleDayConversion();

    std::cout << "\n=== All examples completed ===" << std::endl;
    std::cout << "\nKey takeaways:" << std::endl;
    std::cout << "  - Schedules are specified in DAYS (integers)" << std::endl;
    std::cout << "  - Internally converted to years for simulation (DAYS_IN_YEAR = 365.25)"
              << std::endl;
    std::cout << "  - Lazy evaluation: simulation triggered on first access" << std::endl;
    std::cout << "  - Query by day number, not by year" << std::endl;
    return 0;
}
