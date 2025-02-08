#!/usr/bin/bash
#
# Version 0.0.0.1
#
# Use the script on your own risk.

# Set pattern0.
pattern0="torchvision.transforms.functional_tensor"

# Set pattern1.
pattern1="torchvision.transforms.functional"

# Change directory.
cd ../../..

# Print working directory.
echo -e "$PWD"

# Find file.
fn=$(find -name "degradations.py")

# Make a copy.
cp "${fn}" "${fn}.bak"

# Print file.
echo -e "${fn}"

# Grep pattern.
match=$(cat "${fn}" | grep "${pattern0}")

# Check on match.
if [[ ! ${match} =~ ^"#" ]]; then
    sed -i "s/${pattern0}/${pattern1}/g" "${fn}"
fi

# Exit script.
exit 0
