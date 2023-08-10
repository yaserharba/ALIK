import json
from pathlib import Path


class ClassesInfo:
    @staticmethod
    def getClassNumber(className):
        pathStr = "data/classesInfo.json"
        with open(pathStr) as json_file:
            data = json.load(json_file)
            for key in data:
                if data[key] == className:
                    return key
        data[str(len(data))] = className
        output_file = Path(pathStr)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(pathStr, "w") as outfile:
            json.dump(data, outfile)
        return ClassesInfo.getClassNumber(className)


if __name__ == '__main__':
    className = "Box"
    d = ClassesInfo.getClassNumber(className)
    print(d)
