import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from agent_desk.cli import main as cli_main

        sys.exit(cli_main(sys.argv[1:]))
    # No arguments: the starter's original single-question runner (agent_desk/main.py).
    from agent_desk.main import main

    sys.exit(main())
