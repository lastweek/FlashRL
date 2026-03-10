#!/bin/bash
# Development environment setup for FlashRL

# Consolidate Python bytecode to single location
export PYTHONPYCACHEPREFIX=.cache/pycache

# Add project to Python path
export PYTHONPATH=/Volumes/CaseSensitive/FlashRL:$PYTHONPATH

echo "✓ FlashRL dev environment activated"
echo "  - Bytecode cache: .cache/pycache/"
echo "  - PYTHONPATH set"
