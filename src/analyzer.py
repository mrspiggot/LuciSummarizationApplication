from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from codebleu import calc_codebleu

class EvaluationMetrics:
    def __init__(self):
        self.rouge = Rouge()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.smoothing_function = SmoothingFunction().method1

    def calculate_bleu_score(self, reference, hypothesis):
        reference = [reference.split()]
        hypothesis = hypothesis.split()
        score = corpus_bleu([reference], [hypothesis], smoothing_function=self.smoothing_function)
        return score

    def calculate_rouge_scores(self, reference, hypothesis):
        scores = self.rouge.get_scores(hypothesis, reference, avg=True)
        return scores

    def calculate_semantic_similarity(self, reference, hypothesis):
        reference_embedding = self.model.encode([reference])
        hypothesis_embedding = self.model.encode([hypothesis])
        similarity = cosine_similarity(reference_embedding, hypothesis_embedding)
        return similarity[0][0]

    def calculate_code_bleu_score(self, reference_code, generated_code):
        code_bleu_score = calc_codebleu([reference_code], [generated_code], lang='python', weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        print(f"{code_bleu_score}, {type(code_bleu_score)}")
        return code_bleu_score
