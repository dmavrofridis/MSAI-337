from global_variables import word_indices

def question_7(text):

    #encode the text using word_indeces corpus

    #decode the text using the same corpus


def encode_text_as_int(text):
    intvector = []

    splittext = [token.lower() for token in word_tokenize(text)]

    for word in splittext:
        if word.lower() in word_indices.keys():
            intvector.append(word_indices[word.lower()])
        else:
            intvector.append(-1);

    print(text, "encoded to ", intvector)

    return intvector


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
