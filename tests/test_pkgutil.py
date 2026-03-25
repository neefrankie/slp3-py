import pkgutil
from slp3 import chapter2

if __name__ == '__main__':
    # Try to figure out how namespace package works.
    print(chapter2.__path__)
    print(chapter2.__name__)
    for module_info in pkgutil.iter_modules(chapter2.__path__, prefix=chapter2.__name__ + '.'):
        print(module_info)