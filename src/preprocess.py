import json

def is_chinese(str):
	for uchar in str:
		if not ('\u4e00' <= uchar<='\u9fff'):
			return False
	return True

# 判断一个unicode是否是数字
def is_number(uchar):
    if '\u0030' <= uchar <='\u0039':
        return True
    else:
        return False

# 判断一个unicode是否是英文字母
def is_alphabet(uchar):
    if ('\u0041' <= uchar<='\u005a') or ('\u0061' <= uchar<='\u007a'):
        return True
    else:
        return False

# 判断是否非汉字，数字和英文字符
def is_other(uchar):
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def str_preprocess(str, expression):
    pos = 0
    exp_list = []
    str_list = []
    str_tmp = ''
    while pos < len(str):
        max_len = 0
        exp_tmp = None
        for exp in expression:
            if int(len(str)) - pos < int(len(exp)):
                continue
            if len(exp) <= max_len:
                continue
            if str[pos:(pos+len(exp))] == exp:
                max_len = len(exp)
                exp_tmp = exp
        if max_len > 0:
            if len(str_tmp) > 0:
                str_list.append(str_tmp)
            exp_list.append(exp_tmp)
            pos += max_len
            str_tmp = ''

        else:
            # print('Str_Tmp', type(str_tmp))
            str_tmp = str_tmp + str[pos]
            pos += 1

    if len(str_tmp) > 0:
        str_list.append(str_tmp)

    return str_list, exp_list


class Comment:
    def __init__(self, aid, dataset_dir):
        '''
        aid: the av id of each video
        dataset_dir:  the directory of dataset.
        '''
        self.aid = aid
        self.dataset_dir = dataset_dir
        if not self.dataset_dir.endswith('/'):
            self.dataset_dir += '/'
        self.content = self.read(aid)
        '''
        For example, self.content = [[Comment], [Comment], ...]
        '''
    def dfs(self, root):
        cmts = []
        if root is None:
            return None
        for son in root:
            cmts.append(son['content']['message'])
            tmp = self.dfs(son['replies'])
            if tmp != None: cmts.extend(tmp)
        return cmts

    def read(self, aid):
        with open(self.dataset_dir + aid + '/' + aid + '.cmt', 'r') as f:
            cmt = json.load(f)
        cmts = self.dfs(cmt['data']['replies'])
        return cmts

class Danmuku:
    def __init__(self, aid, dataset_dir):
        '''
        aid: the av id of each video
        dataset_dir:  the directory of dataset.
        '''
        self.aid = aid
        self.dataset_dir = dataset_dir
        if not self.dataset_dir.endswith('/'):
            self.dataset_dir += '/'
        self.content = self.read(aid)
        '''
        For example, self.content = [['danmu', offset in the video, the absolute time published],...]
        '''

    def read(self, aid):
        dan = []
        with open(self.dataset_dir + aid + '/' + aid + '.dan', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                dan.append(dan)


if __name__ == '__main__':
    # cmt = Comment('4704057', '../dataset/')
    # print(cmt.content)
    # dan = Danmuku('4704057', '../dataset/')
