import os
import glob
import functools
import shutil

from git import Repo


class CodeFilePreparation:

    begin_token = "<BOF>\n"
    end_token = "<EOF>\n"

    def __init__(self, github_link, output_folder):
        self._github_link = github_link
        self._output_folder = output_folder

        # where to clone our rep
        pattern = ".com"
        rep_name = self._github_link[self._github_link.index(pattern) + len(pattern) + 1:]
        self._path_to_clone = os.path.join(self._output_folder, rep_name.replace('/', '_'))
        self._path_to_code_file = os.path.join(self._output_folder, rep_name.replace('/', '_') + '.txt')

    def _download(self):
        Repo.clone_from(self._github_link, self._path_to_clone)

    def _merge_py_files(self):

        files = []
        for filename in glob.iglob(os.path.join(self._path_to_clone, '**', '*.py'), recursive=True):
            with open(filename, 'r') as myfile:
                # here we add begin & start tokens for file
                try:
                    code_tmp = myfile.read()

                    # if file not empty
                    if code_tmp:
                        if code_tmp[-1] != '\n':
                            # add new line in the endif it is not the case
                            code_tmp += '\n'
                        code_tmp = CodeFilePreparation.begin_token + code_tmp + CodeFilePreparation.end_token

                        # add his content with BOF / EOF to array
                        files.append(code_tmp)
                except Exception as e:
                    print("exception for file {}".format(filename))
                    print(e)
        # return one big string for all the rep files
        return ''.join(files)

    def _write_to_file(self, code):
        with open(self._path_to_code_file, "w") as text_file:
            text_file.write(code)


    # return path to file
    def prepare(self):
        # prepare only in case if target file is missing
        if not os.path.exists(self._path_to_code_file) or os.stat(self._path_to_code_file).st_size == 0:
            try:
                self._download()
                code = self._merge_py_files()
                self._write_to_file(code)
            except Exception as e:
                print("exception during processing of " +  self._github_link)
                print(e)

        return self._path_to_code_file

def main():
    tmp_output_path = '/tmp/github_code'
    database_folder = r'code_data'

    with open('github_reps.txt', 'r') as f:
        content = f.readlines()

    content = [c.strip() for c in content]

    files = []
    for link in content:
        print("Processing " + link)

        preparator = CodeFilePreparation(link, tmp_output_path)
        code_file = preparator.prepare()

        files.append(code_file)


    if not os.path.exists(database_folder):
        os.mkdir(database_folder)

    for file in files:
        shutil.copy2(file, database_folder)



    pass

if __name__ == "__main__":
    main()