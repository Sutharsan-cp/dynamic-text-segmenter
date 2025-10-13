#ifndef WORKING_TEXT_SEGMENTER_H
#define WORKING_TEXT_SEGMENTER_H

#include <string>
#include <vector>

// Forward declaration for the metrics struct
struct SegmentationMetrics {
    double precision, recall, pk, f1;
    int predictedSegments, trueSegments;
    std::string qualityAssessment;
};

class WorkingTextSegmenter {
public:
    // Constructor
    WorkingTextSegmenter(double mu_val, double sigma_val, double gamma_val, double r_val);

    // Main public methods
    void preprocessText(const std::string& text);
    std::vector<int> segmentText();
    std::vector<std::vector<std::string>> getSegmentedSentences(const std::vector<int>& boundaries);
    SegmentationMetrics evaluateSegmentation(const std::vector<int>& predictedBoundaries,
                                               const std::vector<int>& trueBoundaries,
                                               int totalSentences);
    void printDetailedOutput(const std::string& text, const std::vector<int>& trueBoundaries,
                               const std::string& paramName, const std::vector<double>& params);

private:
    // Private data members
    std::vector<std::string> sentences;
    std::vector<std::vector<std::string>> sentenceWords;
    double mu, sigma, gamma, r;
    std::vector<std::vector<double>> similarityMatrix;
    std::vector<std::vector<double>> S;

    // Private helper methods
    void buildSimilarityMatrix();
    void precomputeSegmentCosts();
    int countTotalWords();
    bool areInSameSegment(int s1, int s2, const std::vector<int>& boundaries);
};

#endif // WORKING_TEXT_SEGMENTER_H