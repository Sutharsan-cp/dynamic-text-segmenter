#include <iostream>
#include <vector>
#include <string>
#include "WorkingTextSegmenter.h" // Include your class header

void runWorkingDemo() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "WORKING TEXT SEGMENTATION DEMO" << std::endl;
    std::cout << "Simple and Effective Approach" << std::endl;
    std::cout << "=============================================================" << std::endl;

    std::string workingText = R"(Artificial intelligence is transforming various industries. Machine learning algorithms are becoming more sophisticated. Deep learning models achieve state of the art results. Natural language processing enables better human computer interaction. Climate change is a pressing global issue. Rising temperatures affect ecosystems worldwide. Renewable energy sources are gaining importance. Carbon emissions need to be reduced significantly. The stock market showed mixed results today. Technology stocks performed well overall. Banking sector faced some challenges. Investors are cautious about future trends. Healthy eating habits contribute to better wellbeing. Regular exercise improves physical fitness. Mental health is equally important for overall wellness. Balanced lifestyle leads to happiness.)";
    std::vector<int> trueBoundaries = {3, 7, 11};

    std::vector<std::vector<double>> parameterSets = {
        {4.0, 1.0, 0.4, 1.0},
        {4.0, 2.0, 0.3, 1.0},
        {4.0, 0.5, 0.5, 1.0}
    };
    std::vector<std::string> paramNames = {
        "Strong Length", "Balanced", "Very Strong Length"
    };

    for (size_t i = 0; i < parameterSets.size(); ++i) {
        WorkingTextSegmenter segmenter(
            parameterSets[i][0],
            parameterSets[i][1],
            parameterSets[i][2],
            parameterSets[i][3]
        );
        segmenter.printDetailedOutput(workingText, trueBoundaries, paramNames[i], parameterSets[i]);
    }

    std::cout << "=============================================================" << std::endl;
    std::cout << "KEY INSIGHTS:" << std::endl;
    std::cout << "- Watch for 'Average similarity' - should be > 0.1" << std::endl;
    std::cout << "- Higher γ values favor segment length consistency" << std::endl;
    std::cout << "- Simple word overlap works better than complex preprocessing" << std::endl;
    std::cout << "=============================================================" << std::endl;
}

int main() {
    runWorkingDemo();
    return 0;
}