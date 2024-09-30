#!/bin/bash

# Step 1: Install stdlib-list if not already installed
pip show stdlib-list >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing stdlib-list..."
    pip install stdlib-list
fi

# Step 2: Get the list of Python 3.10 standard library modules
echo "Getting the list of Python 3.10 built-in modules..."
python3 -c "from stdlib_list import stdlib_list; print('\n'.join(stdlib_list('3.10')))" > stdlib_packages.txt

# Step 3: Get the list of installed packages
echo "Getting the list of installed packages..."
pip freeze > installed_packages.txt

# Step 4: Filter out the built-in modules from the installed packages
echo "Filtering out built-in modules..."
grep -vxFf stdlib_packages.txt installed_packages.txt > manually_installed_packages.txt

# Step 5: Output the list of manually installed packages
echo "List of manually installed packages (excluding Python 3.10 built-ins) saved to manually_installed_packages.txt"
cat manually_installed_packages.txt

# Save this directly to a requirements file
cp manually_installed_packages.txt requirements.txt

# Step 6: Clean up redundant files
echo "Cleaning up temporary files..."
rm stdlib_packages.txt manually_installed_packages.txt installed_packages.txt

echo "Process complete. Manually installed packages saved in 'requirements.txt' and 'manually_installed_packages.txt'."
