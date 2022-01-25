from global_variables import word_indices
from nltk.tokenize import word_tokenize


# not sure this is how he wants it, supposedly these are supposed to be seperate functions,
# but that should be an easy change...
def question_5(text):
    # encode the text using word_indeces corpus
    int_vector = encode_text_as_int(text)
    # decode the text using the same corpus
    recovered_text = decode_ints_as_text(int_vector)

    return recovered_text


def encode_text_as_int(text):
    int_vector = []

    split_text = [token.lower() for token in word_tokenize(text)]

    for word in split_text:
        if word.lower() in word_indices.keys():
            int_vector.append(word_indices[word.lower()])
        else:
            int_vector.append(-1)

    print(text, "encoded to ", int_vector)

    return int_vector


def decode_ints_as_text(int_vector):
    outstring = ""

    for i in int_vector:
        if i != -1:
            # poor man's implementation
            for key, value in word_indices.items():
                if value == i:
                    print(i, " ", value, " ", key)
                    outstring += " "
                    outstring += key
                    continue
        else:
            outstring += " <unk>"

    print(int_vector, "decoded to ", outstring)

    return outstring
