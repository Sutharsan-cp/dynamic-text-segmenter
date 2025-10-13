# Dynamic Text Segmenter

A compact C++ reference implementation of a dynamic text segmenter that balances sentence similarity and a length prior. Designed for demonstration and teaching of segmentation algorithms.

## Key components

- Core implementation: [src/WorkingTextSegmenter.cpp](src/WorkingTextSegmenter.cpp)
- Public interface / API: [include/WorkingTextSegmenter.h](include/WorkingTextSegmenter.h) — see [`WorkingTextSegmenter`](include/WorkingTextSegmenter.h) and [`SegmentationMetrics`](include/WorkingTextSegmenter.h)
- Example runner: [main.cpp](main.cpp)
- Standalone demo: [sample/DAA.cpp](sample/DAA.cpp)
- Prebuilt Windows executable: [segmenter.exe](segmenter.exe)
- Referenced paper: [theory_paper/E03-1058.pdf](theory_paper/E03-1058.pdf)
- License: [LICENSE](LICENSE)

## Overview

The algorithm partitions a sequence of sentences into coherent segments by optimizing a cost that trades off:

- Topic similarity: sentences in the same segment should have high word-overlap similarity.
- Segment length: segments are encouraged to be near a target length using a Gaussian prior.

A dynamic programming solver finds the global optimum segmentation.

## Features

- Simple sentence splitting and tokenization (lowercasing, punctuation removal).
- Sentence-to-sentence similarity using normalized word overlap.
- Dynamic programming segmentation with parameters for length prior and weight between similarity and length.
- Basic evaluation metrics: Precision, Recall, F1, and Pk.

## Parameters

- $ \mu $ — desired/expected segment length (in sentences).
- $ \sigma $ — standard deviation (allowed variability of segment length).
- $ \gamma $ — trade-off weight between length prior and similarity (range 0..1).
- $ r $ — reserved for future use (kept for compatibility).

Tune $ \gamma $ to favor length consistency (higher) or similarity (lower).

## Build

Recommended: g++ (C++17)

```sh
g++ -std=c++17 -Iinclude main.cpp src/WorkingTextSegmenter.cpp -o segmenter
```

Run:

```sh
./segmenter
```

Or use the provided [segmenter.exe](segmenter.exe) on Windows.

## Sample/demo

- The runnable demo is in [main.cpp](main.cpp) and uses the implementation in [src/WorkingTextSegmenter.cpp](src/WorkingTextSegmenter.cpp) via the header [include/WorkingTextSegmenter.h](include/WorkingTextSegmenter.h).
- A standalone sample implementation with the same demo is available at [sample/DAA.cpp](sample/DAA.cpp).

## Quick notes

- Check "Average similarity" printed by the program — values > 0.1 indicate useful sentence overlap for this simple similarity metric.
- Increase γ to favor consistent segment lengths, reduce γ to rely more on similarity.
- This implementation is intentionally simple (word overlap) — replace similarity calculation in [src/WorkingTextSegmenter.cpp](src/WorkingTextSegmenter.cpp) for more advanced behavior.

## How it Works 🧠

The algorithm splits a long text into coherent paragraphs by balancing two main goals:

Topic Similarity: Sentences within the same segment should be about the same topic. The program measures this by calculating the word overlap between sentences.

Segment Length: Segments should be a consistent, desirable length. The program uses a length prior (a target length) and penalizes segments that are too long or too short.

A dynamic programming approach is used to find the optimal segmentation that minimizes a combined "cost" function, considering both similarity and length across the entire text.

## License

- MIT — see [LICENSE](LICENSE).

## References

- Implementation & API: [`WorkingTextSegmenter`](include/WorkingTextSegmenter.h), [`SegmentationMetrics`](include/WorkingTextSegmenter.h)
- Files:
  - [main.cpp](main.cpp)
  - [src/WorkingTextSegmenter.cpp](src/WorkingTextSegmenter.cpp)
  - [include/WorkingTextSegmenter.h](include/WorkingTextSegmenter.h)
  - [sample/DAA.cpp](sample/DAA.cpp)
  - [segmenter.exe](segmenter.exe)
  - [theory_paper/E03-1058.pdf](theory_paper/E03-1058.pdf)
  - [LICENSE](LICENSE)
