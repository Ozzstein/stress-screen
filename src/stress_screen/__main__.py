import multiprocessing

from stress_screen.cli import main

if __name__ == "__main__":
    # Required for frozen (PyInstaller) binaries: dependencies that touch
    # multiprocessing (resource_tracker) respawn sys.executable — which is
    # the CLI itself in a frozen app. freeze_support() intercepts those
    # helper launches instead of re-entering argparse.
    multiprocessing.freeze_support()
    main()
