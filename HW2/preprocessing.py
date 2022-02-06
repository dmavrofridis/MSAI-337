def load_text(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            line.replace("\n", " </s> ").split()
    return lines




def string_to_lower(text):
    text = [i.lower() for i in text]
    return text

def splitting_tokens(text):
    corpus = []
    for lines in text:
        splitted_lines = lines.split(' ')
        corpus.append(splitted_lines)
    return corpus

def lists_to_tokens(text):
    tokens =[]
    for line in text:
        for word in line:
            if word != '' or word != '=' or word != '  ':
                tokens.append(word)
    return tokens







