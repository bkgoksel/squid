import unittest


def main():
    suite: unittest.TestSuite = unittest.defaultTestLoader.discover('test', top_level_dir='.')
    runner: unittest.TestRunner = unittest.runner.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    main()
