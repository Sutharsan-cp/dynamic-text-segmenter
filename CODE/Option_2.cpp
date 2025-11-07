#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Slider.H>
#include <FL/Fl_Box.H>
#include <FL/fl_ask.H>

class TextSegmenter {
private:
    std::vector<std::string> sentences;
    std::vector<std::vector<std::string>> sentenceWords;
    double mu, sigma, gamma, r;

public:
    TextSegmenter(double mu_val, double sigma_val, double gamma_val, double r_val)
        : mu(mu_val), sigma(sigma_val), gamma(gamma_val), r(r_val) {}

    void preprocessText(const std::string& text) {
        sentences.clear();
        sentenceWords.clear();

        std::string cleanText = text;
        // Replace multiple newlines with spaces
        size_t pos = 0;
        while ((pos = cleanText.find('\n', pos)) != std::string::npos) {
            cleanText.replace(pos, 1, " ");
            pos += 1;
        }

        // Simple sentence splitting by period
        size_t start = 0;
        size_t end = cleanText.find('.');

        while (end != std::string::npos) {
            std::string sentence = cleanText.substr(start, end - start);
            sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
            sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);

            if (!sentence.empty() && sentence.length() > 5) {
                sentences.push_back(sentence);

                // Tokenize words
                std::vector<std::string> words;
                std::istringstream wordStream(sentence);
                std::string word;

                while (wordStream >> word) {
                    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                    word.erase(std::remove_if(word.begin(), word.end(),
                             [](char c) { return std::ispunct(c); }), word.end());
                    if (!word.empty() && word.length() > 1) {
                        words.push_back(word);
                    }
                }
                sentenceWords.push_back(words);
            }

            start = end + 1;
            end = cleanText.find('.', start);
        }

        if (start < cleanText.length()) {
            std::string lastSentence = cleanText.substr(start);
            lastSentence.erase(0, lastSentence.find_first_not_of(" \t\n\r"));
            lastSentence.erase(lastSentence.find_last_not_of(" \t\n\r") + 1);
            if (!lastSentence.empty() && lastSentence.length() > 5) {
                sentences.push_back(lastSentence);

                std::vector<std::string> words;
                std::istringstream wordStream(lastSentence);
                std::string word;

                while (wordStream >> word) {
                    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                    word.erase(std::remove_if(word.begin(), word.end(),
                             [](char c) { return std::ispunct(c); }), word.end());
                    if (!word.empty() && word.length() > 1) {
                        words.push_back(word);
                    }
                }
                sentenceWords.push_back(words);
            }
        }
    }

    double calculateSegmentDensity(int start, int end) {
        if (start > end) return 0.0;

        int segmentLength = end - start + 1;
        if (segmentLength <= 0) return 0.0;

        double totalSimilarity = 0.0;
        int comparisons = 0;

        for (int i = start; i <= end; ++i) {
            for (int j = i; j <= end; ++j) {
                if (i == j) {
                    totalSimilarity += 1.0;
                } else {
                    const auto& words1 = sentenceWords[i];
                    const auto& words2 = sentenceWords[j];

                    int commonWords = 0;
                    for (const auto& word : words1) {
                        if (std::find(words2.begin(), words2.end(), word) != words2.end()) {
                            commonWords++;
                        }
                    }

                    int minLength = std::min(words1.size(), words2.size());
                    double similarity = (minLength > 0) ? (double)commonWords / minLength : 0.0;
                    totalSimilarity += similarity;
                }
                comparisons++;
            }
        }

        double density = (comparisons > 0) ? totalSimilarity / comparisons : 0.0;
        double normalizedDensity = density / std::pow(segmentLength, r);

        return normalizedDensity;
    }

    std::vector<int> segmentText() {
        int T = sentences.size();
        if (T == 0) return {};
        if (T == 1) return {-1, 0};

        std::vector<double> C(T, std::numeric_limits<double>::max());
        std::vector<int> prev(T, -1);

        C[0] = 0.0;

        for (int t = 0; t < T; ++t) {
            for (int s = 0; s <= t; ++s) {
                int segmentLength = t - s + 1;

                double lengthCost = gamma * std::pow(segmentLength - mu, 2) / (2.0 * sigma * sigma);
                double density = calculateSegmentDensity(s, t);
                double similarityCost = (1.0 - gamma) * (1.0 - density);

                double previousCost = (s == 0) ? 0.0 : C[s - 1];
                double totalCost = previousCost + lengthCost + similarityCost;

                if (totalCost < C[t]) {
                    C[t] = totalCost;
                    prev[t] = s - 1;
                }
            }
        }

        std::vector<int> boundaries;
        int current = T - 1;

        boundaries.push_back(current);
        while (current >= 0 && prev[current] >= 0) {
            boundaries.push_back(prev[current]);
            current = prev[current];
        }
        boundaries.push_back(-1);

        std::reverse(boundaries.begin(), boundaries.end());
        return boundaries;
    }

    std::vector<std::string> getSegmentedText(const std::vector<int>& boundaries) {
        std::vector<std::string> segments;

        for (int i = 0; i < boundaries.size() - 1; ++i) {
            int start = boundaries[i] + 1;
            int end = boundaries[i + 1];

            if (start <= end) {
                std::string segment;
                for (int j = start; j <= end; ++j) {
                    segment += sentences[j] + ". ";
                }
                segments.push_back(segment);
            }
        }

        return segments;
    }

    std::vector<std::string> segment(const std::string& text) {
        preprocessText(text);
        auto boundaries = segmentText();
        return getSegmentedText(boundaries);
    }
};

class TextSegmenterGUI {
private:
    Fl_Window* window;
    Fl_Text_Display* textDisplay;
    Fl_Text_Buffer* textBuffer;
    Fl_Text_Buffer* styleBuffer;
    Fl_Input* textInput;
    Fl_Slider* muSlider;
    Fl_Slider* sigmaSlider;
    Fl_Slider* gammaSlider;
    Fl_Slider* rSlider;
    Fl_Box* muValue;
    Fl_Box* sigmaValue;
    Fl_Box* gammaValue;
    Fl_Box* rValue;

    TextSegmenter segmenter;

    // Style table for colored text
    Fl_Text_Display::Style_Table_Entry styleTable[8];

public:
    TextSegmenterGUI() : segmenter(4.0, 1.0, 0.4, 1.0) {
        // Initialize style table with different colors
        styleTable[0].color = FL_BLACK;      // Style A - Black
        styleTable[0].font = FL_HELVETICA;
        styleTable[0].size = 12;

        styleTable[1].color = FL_RED;        // Style B - Red
        styleTable[1].font = FL_HELVETICA;
        styleTable[1].size = 12;

        styleTable[2].color = FL_DARK_GREEN; // Style C - Green
        styleTable[2].font = FL_HELVETICA;
        styleTable[2].size = 12;

        styleTable[3].color = FL_BLUE;       // Style D - Blue
        styleTable[3].font = FL_HELVETICA;
        styleTable[3].size = 12;

        styleTable[4].color = FL_MAGENTA;    // Style E - Magenta
        styleTable[4].font = FL_HELVETICA;
        styleTable[4].size = 12;

        styleTable[5].color = FL_CYAN;       // Style F - Cyan
        styleTable[5].font = FL_HELVETICA;
        styleTable[5].size = 12;

        styleTable[6].color = FL_DARK_YELLOW; // Style G - Dark Yellow
        styleTable[6].font = FL_HELVETICA;
        styleTable[6].size = 12;

        styleTable[7].color = FL_DARK3;      // Style H - Dark Gray
        styleTable[7].font = FL_HELVETICA;
        styleTable[7].size = 12;

        createGUI();
    }

    void createGUI() {
        // Create main window
        window = new Fl_Window(800, 700, "Text Segmentation Tool - C++ GUI");

        // Input area
        new Fl_Box(10, 10, 200, 25, "Input Text:");
        textInput = new Fl_Input(10, 35, 780, 100, "");

        // Set default text
        const char* defaultText =
            "Artificial intelligence is transforming various industries. "
            "Machine learning algorithms are becoming more sophisticated. "
            "Deep learning models achieve state of the art results. "
            "Natural language processing enables better human computer interaction. "
            "Climate change is a pressing global issue. Rising temperatures affect ecosystems worldwide. "
            "Renewable energy sources are gaining importance. Carbon emissions need to be reduced significantly. "
            "The stock market showed mixed results today. Technology stocks performed well overall. "
            "Banking sector faced some challenges. Investors are cautious about future trends. "
            "Healthy eating habits contribute to better wellbeing. Regular exercise improves physical fitness. "
            "Mental health is equally important for overall wellness. Balanced lifestyle leads to happiness.";
        textInput->value(defaultText);

        // Parameters
        int y_pos = 150;
        new Fl_Box(10, y_pos, 200, 25, "Parameters:");
        y_pos += 30;

        // Mu parameter
        new Fl_Box(20, y_pos, 100, 25, "μ (Length):");
        muSlider = new Fl_Slider(120, y_pos, 200, 25, "");
        muSlider->type(FL_HORIZONTAL);
        muSlider->bounds(2, 10);
        muSlider->value(4.0);
        muValue = new Fl_Box(330, y_pos, 50, 25, "4.0");
        y_pos += 35;

        // Sigma parameter
        new Fl_Box(20, y_pos, 100, 25, "σ (Deviation):");
        sigmaSlider = new Fl_Slider(120, y_pos, 200, 25, "");
        sigmaSlider->type(FL_HORIZONTAL);
        sigmaSlider->bounds(0.1, 5.0);
        sigmaSlider->value(1.0);
        sigmaValue = new Fl_Box(330, y_pos, 50, 25, "1.0");
        y_pos += 35;

        // Gamma parameter
        new Fl_Box(20, y_pos, 100, 25, "γ (Weight):");
        gammaSlider = new Fl_Slider(120, y_pos, 200, 25, "");
        gammaSlider->type(FL_HORIZONTAL);
        gammaSlider->bounds(0.0, 1.0);
        gammaSlider->value(0.4);
        gammaValue = new Fl_Box(330, y_pos, 50, 25, "0.4");
        y_pos += 35;

        // R parameter
        new Fl_Box(20, y_pos, 100, 25, "r (Density):");
        rSlider = new Fl_Slider(120, y_pos, 200, 25, "");
        rSlider->type(FL_HORIZONTAL);
        rSlider->bounds(0.1, 3.0);
        rSlider->value(1.0);
        rValue = new Fl_Box(330, y_pos, 50, 25, "1.0");
        y_pos += 45;

        // Segment button
        Fl_Button* segmentBtn = new Fl_Button(10, y_pos, 150, 30, "Segment Text");
        segmentBtn->callback(segmentCallback, this);

        // Clear button
        Fl_Button* clearBtn = new Fl_Button(170, y_pos, 150, 30, "Clear");
        clearBtn->callback(clearCallback, this);
        y_pos += 50;

        // Output area
        new Fl_Box(10, y_pos, 200, 25, "Segmented Output:");
        y_pos += 25;

        // Create text display with colored output
        textBuffer = new Fl_Text_Buffer();
        styleBuffer = new Fl_Text_Buffer();

        textDisplay = new Fl_Text_Display(10, y_pos, 780, 400);
        textDisplay->buffer(textBuffer);
        textDisplay->highlight_data(styleBuffer, styleTable, 8, 'A', 0, 0);

        // Update slider callbacks
        muSlider->callback(updateSliderCallback, this);
        sigmaSlider->callback(updateSliderCallback, this);
        gammaSlider->callback(updateSliderCallback, this);
        rSlider->callback(updateSliderCallback, this);

        window->end();
        window->show();
    }

    void run() {
        Fl::run();
    }

    void updateSliderValues() {
        char buffer[10];

        snprintf(buffer, sizeof(buffer), "%.1f", muSlider->value());
        muValue->copy_label(buffer);

        snprintf(buffer, sizeof(buffer), "%.1f", sigmaSlider->value());
        sigmaValue->copy_label(buffer);

        snprintf(buffer, sizeof(buffer), "%.2f", gammaSlider->value());
        gammaValue->copy_label(buffer);

        snprintf(buffer, sizeof(buffer), "%.1f", rSlider->value());
        rValue->copy_label(buffer);

        // Update segmenter parameters
        segmenter = TextSegmenter(muSlider->value(), sigmaSlider->value(),
                                gammaSlider->value(), rSlider->value());
    }

    void segmentText() {
        const char* inputText = textInput->value();
        if (!inputText || strlen(inputText) == 0) {
            fl_alert("Please enter some text to segment.");
            return;
        }

        try {
            auto segments = segmenter.segment(inputText);
            displaySegments(segments);
        } catch (const std::exception& e) {
            fl_alert("Error during segmentation: %s", e.what());
        }
    }

    void displaySegments(const std::vector<std::string>& segments) {
        textBuffer->text("");
        styleBuffer->text("");

        if (segments.empty()) {
            textBuffer->append("No segments found.");
            return;
        }

        std::string fullText;
        std::string styleText;

        for (size_t i = 0; i < segments.size(); ++i) {
            // Add segment header
            std::string header = "=== Segment " + std::to_string(i + 1) + " ===\n";
            fullText += header;

            // Style header (use default style A)
            styleText += std::string(header.length(), 'A');

            // Add segment content with color
            fullText += segments[i] + "\n\n";

            // Style segment content with different colors (B, C, D, etc.)
            char styleChar = 'B' + (i % 7); // Use styles B through H
            styleText += std::string(segments[i].length() + 2, styleChar);
        }

        textBuffer->text(fullText.c_str());
        styleBuffer->text(styleText.c_str());
    }

    void clearText() {
        textBuffer->text("");
        styleBuffer->text("");
    }

    // Static callbacks for FLTK
    static void segmentCallback(Fl_Widget* widget, void* data) {
        TextSegmenterGUI* gui = static_cast<TextSegmenterGUI*>(data);
        gui->segmentText();
    }

    static void clearCallback(Fl_Widget* widget, void* data) {
        TextSegmenterGUI* gui = static_cast<TextSegmenterGUI*>(data);
        gui->clearText();
    }

    static void updateSliderCallback(Fl_Widget* widget, void* data) {
        TextSegmenterGUI* gui = static_cast<TextSegmenterGUI*>(data);
        gui->updateSliderValues();
    }
};

int main() {
    TextSegmenterGUI gui;
    gui.run();
    return 0;
}