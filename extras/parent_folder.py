"""svg3d is just a quick and dirty single-file library, so this is a ghetto
way of importing it from the parent folder for testing purposes."""

from inspect import getsourcefile
import os.path as path, sys
package_path = path.dirname(path.abspath(getsourcefile(lambda:0)))
package_path = package_path[:package_path.rfind(path.sep)]
sys.path.insert(0, package_path)
import svg3d
