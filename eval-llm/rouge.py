from rouge_score import rouge_scorer

target = "target"
prediction = "prediction"

scorer =  rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = scorer.score(target, prediction)

print(scores)
