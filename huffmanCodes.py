import heapq
import numpy as np

class Node:
    def __init__(self, value: str, frequency: int, l_node = None, r_node = None):
        self.left_node, self.right_node = l_node, r_node
        self.value, self.frequency = value, frequency
    
    def __lt__(self, other):
        return self.frequency < other.frequency



class Hufman:
    coding_dic = {}
    uncoding_dic = {}
    text = None
    coded_text = None

    def graph_searching(self, root: Node, binary_code: str):
        if root.value != None:
            self.coding_dic[root.value] = binary_code
            return

        binary_code_left = binary_code + "1"
        self.graph_searching(root.left_node, binary_code_left)

        binary_code_right = binary_code + "0"
        self.graph_searching(root.right_node, binary_code_right)

        

    def generateHuffmanTree(self, text: str):
        self.text = text

        freq_dic = {letter: self.text.count(letter) for letter in set(self.text)}

        self.nodes = [Node(letter, freq) for letter, freq in freq_dic.items()]

        heapq.heapify(self.nodes)

        while len(self.nodes) > 1:
            left, right = heapq.heappop(self.nodes), heapq.heappop(self.nodes)
            freq_sum = left.frequency + right.frequency
            parent_node = Node(None, freq_sum, left, right)
            heapq.heappush(self.nodes, parent_node)
        root = self.nodes[0]
        return root

    def code_text(self):
        self.coded_text =  "".join([self.coding_dic[letter] for letter in self.text])
        return self.coded_text
    
    def uncode_text(self, binary_seq: str, root: Node, uncoded_str: str) -> str:

        self.uncoding_dic = {v: k for k, v in self.coding_dic.items()}

        i = 1
        wynik = ""
        while len(binary_seq) > 0:
            if binary_seq[:i] in self.uncoding_dic.keys():
                wynik += self.uncoding_dic[binary_seq[:i]]
                binary_seq = binary_seq[i:]
                i = 1
            else:
                i+=1
        return wynik

    def save(self): # zapisuje kod oraz zakodowany tekst,
        with open("coded_hufman.txt", "w") as file:
            file.write(self.coded_text)
    
    def load(self): # wczytuje zakodowany tekst oraz kod.
        with open("coded_hufman.txt", "r") as file:
            self.coded_text = file.read()
        # print(self.coded_text)


    def count_entropy(self, probability: np.array) -> int:
        probability = [number/ sum(probability) for number in probability]
        return -np.sum(probability * np.log2(probability), axis=0)


    def count_efectivity(self):
        freq_dic = {letter: self.text.count(letter) for letter in set(self.text)}
        entropy = self.count_entropy(np.array(list(freq_dic.values())))
        print("entropy", entropy)
        print("efectivity",entropy / (sum([len(self.coding_dic[key]) * value for key, value in freq_dic.items()]) / sum(freq_dic.values())))


if __name__ == "__main__":
    hf = Hufman()
    with open("norm_wiki_en.txt", "r") as file:
        norm_wiki = file.read()[:1000]

    root =hf.generateHuffmanTree(norm_wiki)
    hf.graph_searching(root, "")
    print("after graph_searching")

    coded_tekst = hf.code_text()
    print("after code_text")
    # print(coded_tekst)
    hf.save()
    print("after saving")
    uncoded_t = hf.uncode_text(coded_tekst, root, "")
    hf.count_efectivity()

    print(norm_wiki == uncoded_t)



# słownikiem ma być cały czas ciąg bitów, i nie znamy długości ciągów więc można 
# stowrzyć zmienną przechowującą długość cigów i jeśli trzeba zwiększyć to zwiększamy 
# onwerujemy inty na bity przy zapisiwaniu i odczytywaniu 