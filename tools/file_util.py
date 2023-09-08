import os
import tempfile
from pathlib import Path


def get_root_project():
    root_project = Path(__file__).parent.parent.resolve()
    # root_project = Path(os.path.abspath(root_project))
    return root_project

class MyNamedTemporaryFile:
    def __init__(self):
        # self.suffix = suffix
        # self.prefix = prefix
        # self.dir = dir
        # self.file = None
        self.file_path = None

    def __enter__(self):
        # self.file = tempfile.NamedTemporaryFile(suffix=self.suffix, prefix=self.prefix, dir=self.dir, delete=False)
        # self.file_path = self.file.name
        # self.file.close()
        # _,f = tempfile.mkstemp()
        # self.file_path = f
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.file_path = self.file.name
        self.file.close()
        return self.file_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_path and os.path.exists(self.file_path):

            os.remove(self.file_path)

# 使用示例
# with MyNamedTemporaryFile() as temp_file:
    # 在这里使用临时文件temp_file
    # print(f"Temporary file path: {temp_file}")

# 临时文件已经被自动删除
