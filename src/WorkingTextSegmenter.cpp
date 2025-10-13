#include "WorkingTextSegmenter.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <chrono>
#include <iomanip>

WorkingTextSegmenter::WorkingTextSegmenter(double mu_val, double sigma_val, double gamma_val, double r_val)
    : mu(mu_val), sigma(sigma_val), gamma(gamma_val), r(r_val) {}

void WorkingTextSegmenter::preprocessText(const std::string& text) {
    sentences.clear();
    sentenceWords.clear();
    std::vector<std::string> rawSentences;
    std::string currentSentence;

    for (char c : text) {
        if (c == '.' || c == '!' || c == '?') {
            if (!currentSentence.empty()) {
                currentSentence.erase(0, currentSentence.find_first_not_of(" \t\n\r"));
                currentSentence.erase(currentSentence.find_last_not_of(" \t\n\r") + 1);
                if (!currentSentence.empty()) {
                    rawSentences.push_back(currentSentence);
                }
                currentSentence.clear();
            }
        } else {
            currentSentence += c;
        }
    }

    for (const auto& sentence : rawSentences) {
        std::vector<std::string> words;
        std::istringstream sentenceStream(sentence);
        std::string word;
        while (sentenceStream >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            word.erase(std::remove_if(word.begin(), word.end(),
                [](char c) { return std::ispunct(c); }), word.end());
            if (!word.empty()) {
                words.push_back(word);
            }
        }
        if (!words.empty()) {
            sentences.push_back(sentence);
            sentenceWords.push_back(words);
        }
    }
    std::cout << "Preprocessed " << sentences.size() << " sentences with "
              << countTotalWords() << " total words" << std::endl;
}

void WorkingTextSegmenter::buildSimilarityMatrix() {
    int T = sentences.size();
    similarityMatrix.assign(T, std::vector<double>(T, 0.0));
    std::cout << "Building similarity matrix..." << std::endl;

    for (int i = 0; i < T; ++i) {
        for (int j = i; j < T; ++j) {
            if (i == j) {
                similarityMatrix[i][j] = 1.0;
            } else {
                int commonWords = 0;
                for (const auto& word : sentenceWords[i]) {
                    if (std::find(sentenceWords[j].begin(), sentenceWords[j].end(), word) != sentenceWords[j].end()) {
                        commonWords++;
                    }
                }
                double similarity = 0.0;
                int minLength = std::min(sentenceWords[i].size(), sentenceWords[j].size());
                if (minLength > 0) {
                    similarity = (double)commonWords / minLength;
                }
                similarityMatrix[i][j] = similarity;
                similarityMatrix[j][i] = similarity;
            }
        }
    }

    double avgSimilarity = 0.0;
    int count = 0;
    for (int i = 0; i < T; ++i) {
        for (int j = i + 1; j < T; ++j) {
            avgSimilarity += similarityMatrix[i][j];
            count++;
        }
    }
    if (count > 0) {
        avgSimilarity /= count;
        std::cout << "Average similarity: " << std::fixed << std::setprecision(3)
                  << avgSimilarity << " (good: >0.1)" << std::endl;
    }
}

void WorkingTextSegmenter::precomputeSegmentCosts() {
    int T = sentences.size();
    S.assign(T, std::vector<double>(T, 0.0));
    for (int s = 0; s < T; ++s) {
        for (int t = s; t < T; ++t) {
            double totalSimilarity = 0.0;
            int pairCount = 0;
            for (int i = s; i <= t; ++i) {
                for (int j = s; j <= t; ++j) {
                    totalSimilarity += similarityMatrix[i][j];
                    pairCount++;
                }
            }
            if (pairCount > 0) {
                double avgSimilarity = totalSimilarity / pairCount;
                S[s][t] = -avgSimilarity;
            }
        }
    }
}

std::vector<int> WorkingTextSegmenter::segmentText() {
    int T = sentences.size();
    if (T == 0) return {};

    buildSimilarityMatrix();
    precomputeSegmentCosts();

    std::vector<double> C(T, std::numeric_limits<double>::max());
    std::vector<int> Z(T, -1);
    C[0] = 0.0;
    Z[0] = -1;

    for (int t = 0; t < T; ++t) {
        for (int s = 0; s <= t; ++s) {
            double segmentLength = t - s + 1;
            double lengthCost = gamma * std::pow(segmentLength - mu, 2) / (2 * sigma * sigma);
            double similarityCost = (1.0 - gamma) * S[s][t];
            double prevCost = (s == 0) ? 0.0 : C[s - 1];
            double totalCost = prevCost + lengthCost + similarityCost;
            if (totalCost < C[t]) {
                C[t] = totalCost;
                Z[t] = s - 1;
            }
        }
    }

    std::vector<int> boundaries;
    int current = T - 1;
    while (current >= 0) {
        boundaries.push_back(current);
        if (Z[current] < 0) break;
        current = Z[current];
    }
    boundaries.push_back(-1);
    std::reverse(boundaries.begin(), boundaries.end());
    std::cout << "Found " << (boundaries.size() - 1) << " segments" << std::endl;
    return boundaries;
}

std::vector<std::vector<std::string>> WorkingTextSegmenter::getSegmentedSentences(const std::vector<int>& boundaries) {
    std::vector<std::vector<std::string>> segments;
    for (size_t i = 0; i < boundaries.size() - 1; ++i) {
        int start = boundaries[i] + 1;
        int end = boundaries[i + 1];
        std::vector<std::string> segment;
        for (int j = start; j <= end; ++j) {
            segment.push_back(sentences[j]);
        }
        segments.push_back(segment);
    }
    return segments;
}

SegmentationMetrics WorkingTextSegmenter::evaluateSegmentation(const std::vector<int>& predictedBoundaries,
                                               const std::vector<int>& trueBoundaries,
                                               int totalSentences) {
    SegmentationMetrics metrics;
    std::vector<int> predBounds = predictedBoundaries;
    std::vector<int> trueBounds = trueBoundaries;
    if (!predBounds.empty() && predBounds[0] == -1) predBounds.erase(predBounds.begin());
    if (!trueBounds.empty() && trueBounds[0] == -1) trueBounds.erase(trueBounds.begin());

    metrics.predictedSegments = predBounds.size() + 1;
    metrics.trueSegments = trueBounds.size() + 1;

    int truePositives = 0;
    std::unordered_set<int> trueSet(trueBounds.begin(), trueBounds.end());
    for (int predBoundary : predBounds) {
        if (trueSet.count(predBoundary)) {
            truePositives++;
        }
    }

    metrics.precision = (predBounds.size() > 0) ? (double)truePositives / predBounds.size() : (trueBounds.empty() ? 1.0 : 0.0);
    metrics.recall = (trueBounds.size() > 0) ? (double)truePositives / trueBounds.size() : (predBounds.empty() ? 1.0 : 0.0);
    metrics.f1 = (metrics.precision + metrics.recall > 0) ? 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) : 0.0;
    
    int windowSize = std::max(1, totalSentences / (2 * (metrics.trueSegments > 1 ? metrics.trueSegments - 1 : 1) ) );
    int errors = 0;
    int totalPairs = 0;
    for (int i = 0; i < totalSentences - windowSize; ++i) {
        int j = i + windowSize;
        bool predictedSame = areInSameSegment(i, j, predBounds);
        bool trueSame = areInSameSegment(i, j, trueBounds);
        if (predictedSame != trueSame) errors++;
        totalPairs++;
    }

    metrics.pk = (totalPairs > 0) ? (double)errors / totalPairs : 0.0;

    if (metrics.pk == 0.0 && metrics.precision == 1.0 && metrics.recall == 1.0) {
        metrics.qualityAssessment = "PERFECT";
    } else if (metrics.pk < 0.1) {
        metrics.qualityAssessment = "EXCELLENT";
    } else if (metrics.pk < 0.2) {
        metrics.qualityAssessment = "GOOD";
    } else if (metrics.pk < 0.3) {
        metrics.qualityAssessment = "FAIR";
    } else {
        metrics.qualityAssessment = "POOR";
    }
    return metrics;
}

void WorkingTextSegmenter::printDetailedOutput(const std::string& text, const std::vector<int>& trueBoundaries,
                               const std::string& paramName, const std::vector<double>& params) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "--- " << paramName << " ---" << std::endl;
    std::cout << "μ=" << params[0] << ", σ=" << params[1]
              << ", γ=" << params[2] << ", r=" << params[3] << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    preprocessText(text);
    auto boundaries = segmentText();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Boundaries: ";
    for (size_t i = 0; i < boundaries.size(); ++i) {
        std::cout << boundaries[i] << (i < boundaries.size() - 1 ? " -> " : "");
    }
    std::cout << std::endl;

    auto segments = getSegmentedSentences(boundaries);
    std::cout << "\nSegments:" << std::endl;
    for (size_t i = 0; i < segments.size(); ++i) {
        std::cout << "Segment " << (i + 1) << " (" << segments[i].size() << " sentences):" << std::endl;
        for (size_t j = 0; j < segments[i].size(); ++j) {
            std::cout << "  " << (j + 1) << ". " << segments[i][j] << std::endl;
        }
        std::cout << std::endl;
    }

    auto metrics = evaluateSegmentation(boundaries, trueBoundaries, sentences.size());
    std::cout << "Results:" << std::endl;
    std::cout << "  Precision: " << std::fixed << std::setprecision(1) << (metrics.precision * 100) << "%" << std::endl;
    std::cout << "  Recall: " << (metrics.recall * 100) << "%" << std::endl;
    std::cout << "  F1: " << (metrics.f1 * 100) << "%" << std::endl;
    std::cout << "  Pk: " << (metrics.pk * 100) << "%" << std::endl;
    std::cout << "  Segments: " << metrics.predictedSegments << " (true: " << metrics.trueSegments << ")" << std::endl;
    std::cout << "  Quality: " << metrics.qualityAssessment << std::endl;
    std::cout << "  Time: " << duration.count() << "ms" << std::endl;

    if (metrics.qualityAssessment == "PERFECT") {
        std::cout << "🎉 PERFECT SEGMENTATION! 🎉" << std::endl;
    }
}

int WorkingTextSegmenter::countTotalWords() {
    int total = 0;
    for (const auto& words : sentenceWords) {
        total += words.size();
    }
    return total;
}

bool WorkingTextSegmenter::areInSameSegment(int s1, int s2, const std::vector<int>& boundaries) {
    if (boundaries.empty()) return true;
    int seg1 = 0, seg2 = 0;
    for (int boundary : boundaries) {
        if (s1 > boundary) seg1++;
        if (s2 > boundary) seg2++;
    }
    return seg1 == seg2;
}