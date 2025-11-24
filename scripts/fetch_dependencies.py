#!/usr/bin/env python3
"""
Digital-Twin Dependency Fetcher
Fetches dependencies to external directory
"""

import os
import sys
import subprocess
import argparse
import shutil
import stat
from pathlib import Path

# Dependency configuration - using commit hashes
DEPENDENCIES = {
    "nanobind": {
        "url": "https://github.com/wjakob/nanobind.git",
        "commit": "dbe8a3c075f628482cfdf0e9fac1d2939794a6ca"
    },
    "volk": {
        "url": "https://github.com/zeux/volk.git",
        "commit": "4d2dba50ae419d0ad34ef27edcb845b749aaebf4"
    },
    "glm": {
        "url": "https://github.com/g-truc/glm.git",
        "commit": "a583c59e1616a628b18195869767ea4d6faca5f4"
    },
        "entt": {
        "url": "https://github.com/skypjack/entt.git",
        "commit": "8375ee0910183b32832c364d7893d074eada2a0f"
    }
}

def remove_readonly(func, path, excinfo):
    """Handler for removing read-only files on Windows"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def run_command(cmd, cwd=None):
    """Execute shell command"""
    try:
        print(f"> {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False

def fetch_dependency(name, config, external_dir):
    """Fetch single dependency"""
    repo_dir = external_dir / name
    
    if repo_dir.exists():
        print(f"[{name}] exists, checking out commit...")
        
        # Check if we already have the right commit
        current_commit = subprocess.run(
            "git rev-parse HEAD", 
            shell=True, 
            cwd=repo_dir, 
            capture_output=True, 
            text=True
        ).stdout.strip()

        if current_commit == config['commit']:
            print(f"[{name}] already at required commit: {current_commit[:8]}")
            return True

        # Fetch and checkout the required commit
        commands = [
            "git fetch origin",
            f"git checkout {config['commit']}",
            "git submodule update --init --recursive"
        ]
    else:
        print(f"[{name}] Cloning...")
        commands = [
            f"git clone {config['url']} {name}",
            f"git checkout {config['commit']}",
            "git submodule update --init --recursive"
        ]
    
    for cmd in commands:
        cmd_cwd = external_dir if "clone" in cmd else repo_dir
        if not run_command(cmd, cmd_cwd):
            print(f"Failed: {cmd}")
            return False
    
    # Verify
    commit_check = subprocess.run(
        "git rev-parse HEAD", 
        shell=True, 
        cwd=repo_dir, 
        capture_output=True, 
        text=True
    )
    
    final_commit = commit_check.stdout.strip()
    if final_commit == config['commit']:
        print(f"[{name}] successfully checked out at commit: {final_commit[:8]}")
        return True
    else:
        print(f"Failed to checkout correct commit for {name}")
        print(f"  Expected: {config['commit']}")
        print(f"  Got: {final_commit}")
        return False

def clean_dependencies(external_dir):
    """Remove all dependencies"""
    if external_dir.exists():
        print("Cleaning external directory...")
        
        for item in external_dir.iterdir():
            if item.is_dir():
                print(f"  Removing {item.name}...")
                try:
                    shutil.rmtree(item, onerror=remove_readonly)
                except Exception as e:
                    print(f"  Warning: Could not remove {item}: {e}")
                    if sys.platform == "win32":
                        run_command(f'rd /s /q "{item}"')
                    else:
                        run_command(f'rm -rf "{item}"')
        
        print("external directory cleaned")
    else:
        print("external directory does not exist")

def main():
    parser = argparse.ArgumentParser(description="Fetch project dependencies")
    parser.add_argument("--force", action="store_true", help="Force re-clone all dependencies")
    parser.add_argument("--clean", action="store_true", help="Remove all dependencies from external")
    args = parser.parse_args()
    
    # Find project root directory
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    external_dir = root_dir / "external"
    
    print("Digital-Twin Dependency Fetcher")
    print("================================")
    print(f"Project root: {root_dir}")
    print(f"External dir: {external_dir}")
    
    # Clean mode
    if args.clean:
        clean_dependencies(external_dir)
        return 0
    
    # Create external directory
    external_dir.mkdir(exist_ok=True)
    
    # Force remove dependencies
    if args.force:
        print("Force mode: cleaning existing dependencies...")
        clean_dependencies(external_dir)
        external_dir.mkdir(exist_ok=True)
    
    # Fetch dependencies
    success = True
    for dep_name, config in DEPENDENCIES.items():
        if not fetch_dependency(dep_name, config, external_dir):
            success = False
            print(f"Failed to fetch {dep_name}")
    
    print("================================")
    if success:
        print("All dependencies fetched successfully")
        print(f"Dependencies are in: {external_dir}")
        return 0
    else:
        print("Failed to fetch some dependencies")
        print("Try running with --force flag to re-clone")
        return 1

if __name__ == "__main__":
    # Check if running with Python
    if not sys.version_info >= (3, 6):
        print("Error: This script requires Python 3.6 or higher")
        sys.exit(1)
    
    sys.exit(main())