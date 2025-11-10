# Impresso OCR Quality Assessment with Bloom Filters

This repository provides a processing pipeline for assessing OCR quality in digitized newspaper collections within the Impresso project ecosystem. It uses Bloom filters and statistical methods to evaluate text recognition accuracy and generate comprehensive quality metrics.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Build System](#build-system)
- [Quality Assessment Methods](#quality-assessment-methods)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [About Impresso](#about-impresso)

## Overview

This pipeline provides a complete framework for OCR quality assessment that:

- **Evaluates OCR Accuracy**: Uses Bloom filter-based lexical matching to assess text recognition quality
- **Supports Multiple Languages**: Process multilingual collections with language-specific dictionaries
- **Scales Horizontally**: Process data across multiple machines without conflicts
- **Handles Large Datasets**: Efficiently process large collections using S3 and local stamp files
- **Maintains Consistency**: Ensure reproducible results with proper dependency management
- **Integrates with S3**: Seamlessly work with both local files and S3 storage

## File Structure

```
├── README.md                   # This file
├── Makefile                    # Main build configuration
├── .env                        # Environment variables (to be created manually from dotenv.sample)
├── dotenv.sample               # Sample environment configuration
├── Pipfile                     # Python dependencies
├── lib/
│   └── ocrqa_bloom.py          # OCR quality assessment script
├── cookbook/                   # Build system components
│   ├── README.md               # Detailed cookbook documentation
│   ├── setup_ocrqa.mk          # OCR QA-specific setup
│   ├── paths_ocrqa.mk          # Path definitions
│   ├── sync_ocrqa.mk           # Data synchronization
│   ├── processing_ocrqa.mk     # Processing targets
│   └── ...                     # Other cookbook components
├── cookbook-repo-addons/       # Repository-specific extensions
│   ├── config-lb-unknowns.mk   # Luxembourgish unknown words config
│   └── ...                     # Other config files
├── configs/                    # Version-specific configurations
│   ├── config-ocrqa-ocrqa-wp_v1.0.6_v1-0-0.mk
│   └── config-ocrqa-ocrqa-wp_v1.0.6_v1-0-1.mk
└── build.d/                    # Local build directory (auto-created)
```

> **Note on Versioning:** The version of this repository should reflect the latest configuration found in the `configs/` directory. This ensures alignment between the codebase and the processing configurations used for production runs.

## Quick Start

Follow these steps to get started with OCR quality assessment:

### 1. Prerequisites

Ensure you have the required system dependencies installed:

- Python 3.11+
- Make (GNU Make recommended)
- Git with Git LFS
- GNU Parallel
- jq (for aggregations)

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y make git git-lfs parallel coreutils python3 python3-pip jq
```

**macOS:**

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install make git git-lfs parallel coreutils python3 jq
```

### 2. Clone and Setup

1. **Clone the repository:**

   ```bash
   git clone --recursive https://github.com/impresso/impresso-ocrqa-unigram-cookbook.git
   cd impresso-ocrqa-unigram-cookbook
   ```

2. **Install system-level dependencies:**

   ```bash
   # Ubuntu or Debian
   bash cookbook/install_apt.sh

   # macOS (with Homebrew)
   bash cookbook/install_brew.sh
   ```

3. **Configure environment:**

   Before running any processing, configure your environment (see [Configuration](#configuration)):

   ```bash
   cp dotenv.sample .env
   # Edit .env with your S3 credentials and settings
   ```

4. **Install Python dependencies:**

   ```bash
   # Using pipenv (recommended)
   export PIPENV_VENV_IN_PROJECT=enabled
   pipenv install
   ```

5. **Initialize the environment:**

   ```bash
   pipenv shell
   make setup
   ```

   The following steps assume that you have activated the pipenv shell.

### 3. Run a Test

Process a small newspaper to verify everything works:

```bash
# Test with a smaller newspaper first
make newspaper NEWSPAPER=actionfem
```

### 4. Process Full Collection

```bash
# Process entire collection
make collection
```

### Step-by-Step Processing

You can also run individual steps:

1. **Sync data:**

   ```bash
   make sync NEWSPAPER=actionfem
   ```

2. **Run processing:**

   ```bash
   make processing-target NEWSPAPER=actionfem
   ```

3. **Upload results:**

   ```bash
   make sync-output NEWSPAPER=actionfem
   ```

## Configuration

### Environment Variables

Edit your `.env` file with these required settings:

```bash
# S3 Configuration (required)
SE_ACCESS_KEY=your_s3_access_key
SE_SECRET_KEY=your_s3_secret_key
SE_HOST_URL=https://os.zhdk.cloud.switch.ch/

# Logging Configuration (optional)
LOGGING_LEVEL=INFO
```

Or provide these variables in your shell environment by other means.

### Important Processing Variables

These can be set in `.env` or passed as command arguments:

- `NEWSPAPER`: Target newspaper to process
- `BUILD_DIR`: Local build directory (default: `build.d`)
- `PARALLEL_JOBS`: Maximum number of parallel years of a newspaper to process
- `COLLECTION_JOBS`: Number of newspaper titles to be run in parallel
- `NEWSPAPER_YEAR_SORTING`: Processing order of years (`shuf` for random, `cat` for chronological)

### S3 Bucket Configuration

Configure S3 buckets in your paths file:

- `S3_BUCKET_REBUILT`: Input data bucket (default: `22-rebuilt-final`)
- `S3_BUCKET_OCRQA`: Output data bucket (default: `140-processed-data-sandbox`)

## Build System

### Core Targets

After installation, these are the main commands you'll use:

- `make help`: Show available targets and current configuration
- `make setup`: Initialize environment (run once after installation)
- `make newspaper`: Process single newspaper
- `make collection`: Process multiple newspapers in parallel
- `make all`: Complete processing pipeline with data sync

### Data Management

- `make sync`: Sync input and output data
- `make sync-input`: Download input data from S3
- `make sync-output`: Upload results to S3 (will never overwrite existing data)
- `make clean-build`: Remove build directory

### Parallel Processing

The system automatically detects CPU cores and configures parallel processing:

```bash
# Process collection with custom parallelization
make collection COLLECTION_JOBS=4 MAX_LOAD=8
```

### Build System Architecture

The build system uses:

- **Stamp Files**: Track processing state without downloading full datasets
- **S3 Integration**: Direct processing from/to S3 storage
- **Distributed Processing**: Multiple machines can work independently
- **Dependency Management**: Automatic dependency resolution via Make

For detailed build system documentation, see [cookbook/README.md](cookbook/README.md).

## Quality Assessment Methods

### Available Methods

The OCR quality assessment script supports multiple evaluation methods:

#### 1. `unk_type_ratio` (Default)

Calculates the ratio of known unique subtoken types to all unique subtoken types. This provides a measure of how many unique words in the text are recognized by the Bloom filter, serving as an indicator of OCR quality.

#### 2. `unk_ratio`

Measures the overall ratio of unknown tokens to total tokens in the document.

### Bloom Filter Dictionaries

The system uses Bloom filters for efficient lexical matching across multiple languages:

- Support for multiple language-specific dictionaries
- Efficient memory usage for large lexicons
- Fast lookup operations
- Configurable via Hugging Face Hub references or local files

## Advanced Usage

### Command-Line Script

To run the OCR quality assessment script, use the following command:

````sh

To run the OCR quality assessment script directly, use the following command:

```bash
python lib/ocrqa_bloom.py \
  --input input.jsonl \
  --bloomdicts bloom1.bloom bloom2.bloom \
  --languages en fr \
  --methods slc unk_ratio \
  --output results.jsonl \
  --lid langident.json
````

### Command-Line Options

#### Input/Output Options

- `--input`: Input JSONL files (default: stdin)
- `--output`: Output file (default: stdout)
- `--log-file FILE`: Write log to FILE

#### Language and Dictionary Options

- `-l, --languages`: Language ISO 2-letter codes (must match the sequence of provided bloom dictionaries)
- `--bloomdicts`: Paths to JSON files containing bloom dictionaries or Hugging Face Hub references (e.g., `hf://model_id/bloom.bloom`)
- `--lid`: Path to language identification file

#### Processing Options

- `--methods`: OCR QA methods to use (default: `unk_type_ratio`)
  - Available: `unk_type_ratio`, `unk_ratio`
- `--keep-best`: Keep only the highest OCR value for a given content item using the first method in `--methods`
- `--unicode-normalization`: Unicode normalization form to apply to input text (default: NFKC)

#### Cost Parameters

- `-C, --single_letter_cost`: Cost for an infrequent single char (default: 0.7)
- `-S, --single_symbol_cost`: Cost for an infrequent symbol char (default: 0.3)

#### Logging Options

- `--log-level`: Logging level (default: INFO)
- `-q, --quiet`: Do not print status messages to stderr
- `-v, --verbose-output`: Print verbose output information

#### S3 Integration Options

- `--s3-output-path`: S3 path to upload the output file after processing or check if it already exists
- `--quit-if-s3-output-exists`: Quit if the output file already exists in the specified S3 bucket
- `--keep-timestamp-only`: After uploading to S3, keep only the timestamp of the local output file for data efficiency
- `--s3-output-dry-run`: Dry run which suppresses all write operations to S3 and checks whether output files exist

### Example Usage

**Basic processing with Hugging Face dictionaries:**

```bash
python lib/ocrqa_bloom.py \
  --input input.jsonl \
  --bloomdicts hf://impresso/bloom-en hf://impresso/bloom-fr \
  --languages en fr \
  --methods unk_type_ratio slc \
  --output results.jsonl \
  --lid langident.json
```

**Processing with S3 integration:**

```bash
python lib/ocrqa_bloom.py \
  --input input.jsonl \
  --bloomdicts hf://impresso/bloom-en \
  --languages en \
  --s3-output-path s3://bucket/path/output.jsonl \
  --quit-if-s3-output-exists
```

### Extended Output for Unknown Words

To produce extended output identifying unknown words (useful for dictionary improvement):

```bash
# For a newspaper in Luxembourgish
CONFIG_LOCAL_MAKE=cookbook-repo-addons/config-lb-unknowns.mk make all NEWSPAPER=your_newspaper
```

### Historical Luxembourgish: Apostrophe Usage After Vowels

This section provides context for understanding OCR challenges in historical Luxembourgish texts.

#### Function of the Apostrophe

- **Indicating long or stressed vowels**
  - _gro'ss_ → modern _grouss_
  - _se'er_ → modern _seier_
- **Marking elision or glottalization**
  - _ge'nt_, _go'f_, _go'w_ (possible sound loss or separation)
- **Clarifying pronunciation in loanwords**
  - _Unio'n_, _situatio'n_, _millio'nen_
- **Separating prefixes or morphemes**
  - _ne'deg_ → modern _néideg_
  - _we'neg_ → modern _wéineg_

#### Spelling Reforms and the Apostrophe

- **Pre-1946**: Apostrophes were common after vowels, often inconsistently
- **1946 Reform**: Reduced apostrophe use, favoring phonetic spelling
- **1975 Reform**: Further simplification, removing unnecessary markers
- **1999 Reform**: Apostrophes after vowels were eliminated, except in contractions (e.g., _d'Kanner_ remains, but _se'er_ → _seier_)

#### Summary

The historical use of apostrophes after vowels served as a **pronunciation guide** for vowel length, stress, and borrowed words. Over time, Luxembourgish orthography **standardized and simplified**, leading to the apostrophe's removal in these contexts. This historical variation presents unique challenges for OCR quality assessment of historical Luxembourgish newspapers.

## Contributing

We welcome contributions to improve this OCR quality assessment pipeline:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test with `make newspaper NEWSPAPER=actionfem`
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Submit a pull request

## Contact

For any questions or issues, please contact [simon.clematide@uzh.ch](mailto:simon.clematide@uzh.ch).

## About Impresso

````

## About Impresso

### Impresso Project

### Default Method

The default method used for OCR quality assessment is `unk_type_ratio`. This method calculates the ratio of known unique subtoken types to all unique subtoken types. It provides a measure of how many unique words in the text are recognized by the Bloom filter, which can be an indicator of OCR quality.

### Example

```sh
python lib/ocrqa_bloom.py --input input.jsonl --bloomdicts hf://model_id/bloom1.bloom hf://model_id/bloom2.bloom --languages en fr --methods slc unk_ratio --output results.jsonl --lid langident.json
````

## Apostrophe Usage After Vowels in Historical Luxembourgish

### **1. Function of the Apostrophe**

- **Indicating long or stressed vowels**
  - _gro’ss_ → modern _grouss_
  - _se’er_ → modern _seier_
- **Marking elision or glottalization**
  - _ge’nt_, _go’f_, _go’w_ (possible sound loss or separation)
- **Clarifying pronunciation in loanwords**
  - _Unio’n_, _situatio’n_, _millio’nen_
- **Separating prefixes or morphemes**
  - _ne’deg_ → modern _néideg_
  - _we’neg_ → modern _wéineg_

### **2. Spelling Reforms and the Apostrophe**

- **Pre-1946**: Apostrophes were common after vowels, often inconsistently.
- **1946 Reform**: Reduced apostrophe use, favoring phonetic spelling.
- **1975 Reform**: Further simplification, removing unnecessary markers.
- **1999 Reform**: Apostrophes after vowels were eliminated, except in contractions (e.g., _d’Kanner_ remains, but _se’er_ → _seier_).

### **3. Summary**

The historical use of apostrophes after vowels served as a **pronunciation guide** for vowel length, stress, and borrowed words. Over time, Luxembourgish orthography **standardized and simplified**, leading to the apostrophe's removal in these contexts.

## Produce Extended output of Finding Unknown Words

```sh
# for a newspaper and Luxembourgish
CONFIG_LOCAL_MAKE=cookbook-repo-addons/config-lb-unknowns.mk  make all
```

## Contact

For any questions or issues, please contact
[simon.clematide@uzh.ch](mailto:simon.clematide@uzh.ch).

## About

### Impresso Project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders.

The project is funded by:

- Swiss National Science Foundation (grants [CRSII5_173719](http://p3.snf.ch/project-173719) and [CRSII5_213585](https://data.snf.ch/grants/grant/213585))
- Luxembourg National Research Fund (grant 17498891)

### Copyright

Copyright (C) 2018-2025 The Impresso team.

Contributors to this program include: [Maud Ehrmann](https://github.com/e-maud), [Simon Clematide](https://github.com/siclemat)

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>
