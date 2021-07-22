import sys

if __name__ == "__main__":
    _ckpt_name = sys.argv[1]
    _test_name = sys.argv[2]
    for line in sys.stdin:
        try:
            _, ckptname, testname, bleutype, score = line.strip().split('\t')
            if ckptname == _ckpt_name and testname == _test_name and score.isdigit():
                print(True)
                exit(0)
        except Exception:
            pass
    print(False)
