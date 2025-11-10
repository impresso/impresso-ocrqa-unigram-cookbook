# Code to sample documents with good OCR
# Usage: make sample-ocr


sample-files: $(BUILD_DIR)/sampling/ocrqa-highscores-de.jsonl \
			  $(BUILD_DIR)/sampling/ocrqa-highscores-fr.jsonl 
#			  $(BUILD_DIR)/sampling/ocrqa-highscores-lb.jsonl

sampled-files: $(BUILD_DIR)/sampling/ocrqa-highscores-data-de.jsonl.bz2 \
$(BUILD_DIR)/sampling/ocrqa-highscores-data-fr.jsonl.bz2

$(BUILD_DIR)/sampling/ocrqa-highscores-%.jsonl:
	@mkdir -p $(dir $@)
	python3 cookbook/lib/s3_sampler.py \
		--s3-prefix s3://42-processed-data-final/ocrqa/ocrqa-ocrqa-wp_v1.0.6_v1-0-1/ \
		--output $@ \
		--filter-expr 'select(.ocrqa_unk_type_ratio > 0.97 and .lg == "$*" and .subtokens > 100 and .subtoken_char_ratio < 0.2)' \
		--transform-expr '{ci_id: .ci_id, lg: .lg, ocrqa: .ocrqa_unk_type_ratio, subtokens:.subtokens,subtoken_char_ratio:.subtoken_char_ratio}' \
		--group-by-expr '.ci_id | split("-") | .[0] + "-" + .[1]' \
		--max-samples-per-group 20 \
		--record-id-field ci_id \
		--random-seed 42 \
		--log-level $(LOGGING_LEVEL) \
		--log-file $@.log.gz
		


$(BUILD_DIR)/sampling/ocrqa-highscores-data-%.jsonl.bz2: $(BUILD_DIR)/sampling/ocrqa-highscores-%.jsonl
	python3 cookbook/lib/s3_compiler.py \
		--input-file $< \
		-o $@ \
		--id-field ci_id \
		--include-from-input ci_id ocrqa lg subtokens subtoken_char_ratio \
		--transform-expr '{text: .ft}' \
		--s3-prefix s3://22-rebuilt-final/ \
		--log-level $(LOGGING_LEVEL) \
		--log-file $@.log.gz
