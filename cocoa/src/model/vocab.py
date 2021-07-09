# TODO: use named tuple to represent entities?
def is_entity(word):
    if not isinstance(word, str):
        return True
    return False


class Vocabulary(object):

    UNK = '<unk>'

    def __init__(self, offset_word=0, offset_char=0, unk=True):
        self.word_to_ind = {}
        self.ind_to_word = {}
        self.char_to_ind = {}
        self.ind_to_char = {}
        self.size_word = 0
        self.size_char = 0
        self.offset_word = offset_word
        self.offset_char = offset_char
        if unk:
            self.add_word(self.UNK)

    def add_words(self, words):
        for w in words:
            self.add_word(w)

    def has_word(self, word):
        return word in self.word_to_ind

    def add_word(self, word):
        if not self.has_word(word):
            ind = self.size_word + self.offset_word
            self.word_to_ind[word] = ind
            self.ind_to_word[ind] = word
            self.size_word += 1

    def to_ind_word(self, word):
        if word in self.word_to_ind:
            return self.word_to_ind[word]
        else:
            # NOTE: if UNK is not enabled, it will throw an exception
            if self.UNK in self.word_to_ind:
                return self.word_to_ind[self.UNK]
            else:
                raise KeyError(str(word))

    def to_word(self, ind):
        return self.ind_to_word[ind]

    def dump_word(self):
        for i, w in self.ind_to_word.items():
            print('{:<8}{:<}'.format(i, w))

    ###
    def has_char(self, char):
        return char in self.char_to_ind

    def add_chars(self, chars):
        for c in chars:
            self.add_char(c)

    def add_char(self, char):
        if not self.has_char(char):
            ind = self.size_char + self.offset_char
            self.char_to_ind[char] = ind
            self.ind_to_char[ind] = char
            self.size_char += 1

    def to_ind_char(self, char):
        if char in self.char_to_ind:
            return self.char_to_ind[char]
        else:
            # NOTE: if UNK is not enabled, it will throw an exception
            if self.UNK in self.char_to_ind:
                return self.char_to_ind[self.UNK]
            else:
                raise KeyError(str(char))

    def to_char(self, ind):
        return self.ind_to_char[ind]

    def dump_char(self):
        for i, w in self.ind_to_char.items():
            print('{:<8}{:<}'.format(i, w))
