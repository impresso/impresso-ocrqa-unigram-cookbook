# Description: Makefile for OCR quality assessment for newspapers
# Read the README.md for more information on how to use this Makefile.
# Or run `make` for online help.

#### ENABLE LOGGING FIRST
# USER-VARIABLE: LOGGING_LEVEL
# Defines the logging level for the Makefile.

# Load our make logging functions
include cookbook/log.mk


# USER-VARIABLE: CONFIG_LOCAL_MAKE
# Defines the name of the local configuration file to include.
#
# This file is used to override default settings and provide local configuration. If a
# file with this name exists in the current directory, it will be included. If the file
# does not exist, it will be silently ignored. Never add the file called config.local.mk
# to the repository! If you have stored config files in the repository set the
# CONFIG_LOCAL_MAKE variable to a different name.
CONFIG_LOCAL_MAKE ?= config.local.mk

# Load local config if it exists (ignore silently if it does not exists)
-include $(CONFIG_LOCAL_MAKE)


# Now we can use the logging function to show the current logging level
  $(call log.info, LOGGING_LEVEL)


#: Show help message
help::
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  setup                 # Prepare the local directories"
	@echo "  collection            # Call make all for each newspaper found in the file $(NEWSPAPERS_TO_PROCESS_FILE)"
	@echo "  all                   # Resync the data from the S3 bucket to the local directory and process all years of a single newspaper"
	@echo "  newspaper             # Process a single newspaper for all years"
	@echo "  sync                  # Sync the data from the S3 bucket to the local directory"
	@echo "  resync                # Remove the local synchronization file stamp and sync again."
	@echo "  clean-build           # Remove the entire build directory"
	@echo "  clean-newspaper       # Remove the local directory for a single newspaper"
	@echo "  help                  # Show this help message"

# Default target when no target is specified on the command line
.DEFAULT_GOAL := help
.PHONY: help


# SETTINGS FOR THE MAKE PROGRAM
include cookbook/make_settings.mk

# If you need to use a different shell than /bin/dash, overwrite it here.
# SHELL := /bin/bash


# SETTINGS FOR THE BUILD PROCESS

# Set the number of parallel launches of newspapers (uses xargs)
# Note: For efficient parallelization the number of cores should be COLLECTION_JOBS * NEWSPAPER_JOBS
#COLLECTION_JOBS ?= 1
#  $(call log.debug, COLLECTION_JOBS)

# Set the number of parallel jobs of newspaper-year files to process
#  $(call log.debug, NEWSPAPER_JOBS)
#NEWSPAPER_JOBS ?= 1 


# SETUP SETTINGS AND TARGETS
include cookbook/setup.mk
include cookbook/setup_python.mk
include cookbook/setup_ocrqa.mk

# Load newspaper list configuration and processing rules
include cookbook/newspaper_list.mk


# SETUP PATHS
include cookbook/paths_canonical.mk
include cookbook/paths_rebuilt.mk
include cookbook/paths_langident.mk
include cookbook/paths_ocrqa.mk


# MAIN TARGETS
include cookbook/main_targets.mk


# SYNCHRONIZATION TARGETS
include cookbook/sync.mk
include cookbook/sync_canonical.mk
include cookbook/sync_rebuilt.mk
include cookbook/sync_langident.mk
include cookbook/sync_ocrqa.mk

include cookbook/clean.mk


# PROCESSING TARGETS
include cookbook/processing.mk
include cookbook/processing_ocrqa.mk



# FURTHER ADDONS
include cookbook-repo-addons/sampling-good-articles.mk
