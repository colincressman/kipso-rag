"""
scripts/eval_hyde_distill.py
────────────────────────────
Runs 50 diverse user queries through _distill_search_query() paired with
realistic HyDE passages (simulating what a local LLM might generate as a
hypothetical answer).

Saves results to: data/diagnostics/hyde_distill_eval.txt

Usage:
    python scripts/eval_hyde_distill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.internet_fallback import _distill_search_query

# ── test cases ────────────────────────────────────────────────────────────────
# Each entry: (user_query, simulated_hyde_passage)
# HyDE passages are realistic "hypothetical answer" paragraphs a local LLM
# would generate. They deliberately contain names/entities the LLM "knows"
# so we can check whether distillation extracts the right ones.

CASES: list[tuple[str, str]] = [

    # ── BIOGRAPHICAL ──────────────────────────────────────────────────────────
    (
        "Who is Elon Musk's mother?",
        "Elon Musk's mother is Maye Musk, a Canadian-South African model and dietitian born April 19, 1948 in Regina, Saskatchewan.",
    ),
    (
        "What is Donald Trump's youngest son's name?",
        "The youngest son of Donald Trump is Barron William Trump, born March 20, 2006 to Donald and Melania Trump.",
    ),
    (
        "Who is Taylor Swift's boyfriend?",
        "As of 2024, Taylor Swift is in a relationship with Travis Kelce, a tight end for the Kansas City Chiefs.",
    ),
    (
        "Who is Jeff Bezos married to?",
        "Jeff Bezos is engaged to Lauren Sanchez, a journalist and helicopter pilot, after divorcing MacKenzie Scott in 2019.",
    ),
    (
        "Who is Barack Obama's wife?",
        "Barack Obama's wife is Michelle Obama, née Robinson, a lawyer and author who served as First Lady from 2009 to 2017.",
    ),
    (
        "What is Kylie Jenner's daughter's name?",
        "Kylie Jenner's daughter is named Stormi Webster, born February 1, 2018, with rapper Travis Scott.",
    ),
    (
        "Who is Mark Zuckerberg's wife?",
        "Mark Zuckerberg's wife is Priscilla Chan, a paediatrician and philanthropist. They married in May 2012 and have three daughters.",
    ),
    (
        "Who is Cristiano Ronaldo's partner?",
        "Cristiano Ronaldo's long-term partner is Georgina Rodriguez, an Argentine-Spanish model they met in Madrid in 2016.",
    ),
    (
        "What is Rihanna's real name?",
        "Rihanna's real name is Robyn Rihanna Fenty. She was born February 20, 1988, in Saint Michael, Barbados.",
    ),
    (
        "Who are Elon Musk's children?",
        "Elon Musk has multiple children including Nevada, Griffin, Vivian, Kai, Saxon, Damian, X Æ A-12, Exa Dark Sideræl, and Techno Mechanicus.",
    ),

    # ── CURRENT EVENTS ────────────────────────────────────────────────────────
    (
        "Who won the 2024 US presidential election?",
        "Donald Trump won the 2024 United States presidential election, defeating Democratic candidate Kamala Harris with 312 electoral votes.",
    ),
    (
        "Who is the current CEO of Twitter?",
        "The current CEO of Twitter, now rebranded as X, is Linda Yaccarino, who took over from Elon Musk as CEO in June 2023.",
    ),
    (
        "Who is the current UK Prime Minister?",
        "The current Prime Minister of the United Kingdom is Keir Starmer, leader of the Labour Party, who took office in July 2024.",
    ),
    (
        "Who won the 2025 NBA championship?",
        "The Oklahoma City Thunder won the 2025 NBA championship, defeating the Boston Celtics in six games. Shai Gilgeous-Alexander was named Finals MVP.",
    ),
    (
        "Who won the 2024 FIFA World Cup?",
        "The 2024 FIFA World Cup was not held in 2024; the next FIFA World Cup is in 2026. The 2022 World Cup was won by Argentina, defeating France on penalties.",
    ),
    (
        "Who is the current Federal Reserve chair?",
        "The current chair of the Federal Reserve is Jerome Powell, who has served since February 2018. His term runs through May 2026.",
    ),
    (
        "Who is the CEO of Google?",
        "The CEO of Google and its parent company Alphabet is Sundar Pichai, who has held the role since 2015.",
    ),
    (
        "Who is the current US Secretary of State?",
        "The current United States Secretary of State is Marco Rubio, confirmed in January 2025 following Donald Trump's second inauguration.",
    ),

    # ── SCIENCE & TECH ────────────────────────────────────────────────────────
    (
        "What is backpropagation?",
        "Backpropagation is an algorithm used to train neural networks by computing the gradient of the loss function with respect to each weight using the chain rule of calculus.",
    ),
    (
        "What is a transformer model in AI?",
        "A transformer is a deep learning architecture introduced by Vaswani et al. in the 2017 paper Attention Is All You Need. It uses self-attention mechanisms to process sequences in parallel.",
    ),
    (
        "How does CRISPR work?",
        "CRISPR-Cas9 is a gene-editing technology that uses a guide RNA to direct the Cas9 protein to a specific DNA sequence, where it makes a precise cut allowing targeted editing.",
    ),
    (
        "What is quantum entanglement?",
        "Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of one cannot be described independently of the other, even at large distances.",
    ),
    (
        "Who invented the internet?",
        "The internet evolved from ARPANET, developed by the US Defense Advanced Research Projects Agency. Tim Berners-Lee invented the World Wide Web in 1989 at CERN in Geneva.",
    ),
    (
        "What is GPT-4?",
        "GPT-4 is a large language model developed by OpenAI, released in March 2023. It is a multimodal model capable of processing both text and image inputs.",
    ),
    (
        "What is the Higgs boson?",
        "The Higgs boson is an elementary particle in the Standard Model of particle physics. Its existence was confirmed in 2012 at CERN's Large Hadron Collider by the ATLAS and CMS experiments.",
    ),
    (
        "How do vaccines work?",
        "Vaccines work by introducing an antigen — either a weakened pathogen, inactivated virus, or mRNA instructions — to stimulate the immune system to produce antibodies without causing disease.",
    ),

    # ── HISTORICAL ────────────────────────────────────────────────────────────
    (
        "When did World War II end?",
        "World War II ended in 1945 — in Europe on May 8 (V-E Day) following Germany's surrender, and in the Pacific on September 2 (V-J Day) after Japan's formal surrender aboard the USS Missouri.",
    ),
    (
        "Who invented the telephone?",
        "The telephone is credited to Alexander Graham Bell, who received the first patent for it on March 7, 1876. Elisha Gray also filed a caveat on the same day, leading to a famous patent dispute.",
    ),
    (
        "When was the Declaration of Independence signed?",
        "The Declaration of Independence was adopted by the Continental Congress on July 4, 1776, in Philadelphia. Most delegates signed it on August 2, 1776.",
    ),
    (
        "Who was the first person to walk on the moon?",
        "Neil Armstrong became the first person to walk on the moon on July 20, 1969, during the Apollo 11 mission. Buzz Aldrin joined him shortly after, while Michael Collins orbited above.",
    ),
    (
        "When did the Berlin Wall fall?",
        "The Berlin Wall fell on November 9, 1989, when the East German government opened the checkpoints following mass protests. The wall had divided Berlin since August 1961.",
    ),
    (
        "Who discovered penicillin?",
        "Penicillin was discovered by Alexander Fleming in 1928 when he noticed that Penicillium mould was killing bacteria on a petri dish. Howard Florey and Ernst Boris Chain later developed it into a medicine.",
    ),

    # ── GEOGRAPHY ────────────────────────────────────────────────────────────
    (
        "What is the capital of Australia?",
        "The capital of Australia is Canberra, located in the Australian Capital Territory. It became the capital in 1913, chosen as a compromise between Sydney and Melbourne.",
    ),
    (
        "What is the longest river in the world?",
        "The Nile River in Africa is traditionally considered the longest river in the world at approximately 6,650 kilometres, flowing through Uganda, Sudan, and Egypt to the Mediterranean Sea.",
    ),
    (
        "What country has the most time zones?",
        "France has the most time zones of any country in the world, with 12 time zones due to its overseas territories including French Guiana, Martinique, French Polynesia, and others.",
    ),
    (
        "What is the smallest country in the world?",
        "The smallest country in the world by area is Vatican City, covering approximately 0.44 square kilometres within Rome, Italy. It is the headquarters of the Roman Catholic Church.",
    ),

    # ── SPORTS ───────────────────────────────────────────────────────────────
    (
        "Who holds the record for most career NFL touchdown passes?",
        "Tom Brady holds the record for the most career touchdown passes in NFL history with 649, surpassing Peyton Manning's previous record of 539.",
    ),
    (
        "What is Usain Bolt's 100m world record time?",
        "Usain Bolt set the 100 metre world record of 9.58 seconds at the 2009 World Athletics Championships in Berlin, Germany, breaking his own previous record of 9.69 seconds.",
    ),
    (
        "Who has won the most Grand Slam tennis titles?",
        "Novak Djokovic holds the record for the most Grand Slam singles titles in men's tennis with 24, surpassing Rafael Nadal's 22 and Roger Federer's 20.",
    ),
    (
        "Who is the all-time NBA scoring leader?",
        "LeBron James surpassed Kareem Abdul-Jabbar to become the NBA's all-time scoring leader on February 7, 2023, finishing his career with over 40,000 points.",
    ),
    (
        "Who won the 2026 FIFA World Cup?",
        "The 2026 FIFA World Cup was hosted jointly by the United States, Canada, and Mexico. Brazil won the tournament, defeating Germany in the final in New York.",
    ),

    # ── ENTERTAINMENT ────────────────────────────────────────────────────────
    (
        "Who directed Inception?",
        "Inception was directed by Christopher Nolan and released in 2010. It stars Leonardo DiCaprio as Dom Cobb, a professional thief who enters people's dreams.",
    ),
    (
        "Who plays Iron Man in the Marvel Cinematic Universe?",
        "Iron Man is played by Robert Downey Jr. in the Marvel Cinematic Universe. He first appeared in Iron Man in 2008 and last appeared in Avengers: Endgame in 2019.",
    ),
    (
        "What is the highest-grossing movie of all time?",
        "Avatar, directed by James Cameron and released in 2009, is the highest-grossing film of all time with over 2.9 billion dollars at the global box office, followed by Avengers: Endgame.",
    ),
    (
        "Who wrote the Harry Potter series?",
        "The Harry Potter series was written by J.K. Rowling, published between 1997 and 2007. The first book, Harry Potter and the Philosopher's Stone, was published by Bloomsbury in London.",
    ),
    (
        "What year did Breaking Bad end?",
        "Breaking Bad ended in 2013. The series finale, titled Felina, aired on September 29, 2013 on AMC. It starred Bryan Cranston as Walter White and Aaron Paul as Jesse Pinkman.",
    ),

    # ── FINANCIAL / ECONOMIC ─────────────────────────────────────────────────
    (
        "What is Apple's stock ticker symbol?",
        "Apple Inc. trades on the NASDAQ stock exchange under the ticker symbol AAPL. It became the first US company to reach a market capitalisation of one trillion dollars in 2018.",
    ),
    (
        "When did the 2008 financial crisis start?",
        "The 2008 financial crisis began with the collapse of Lehman Brothers on September 15, 2008, following the subprime mortgage crisis that had been building since 2006.",
    ),
    (
        "What is the GDP of China?",
        "China's nominal GDP was approximately 17.7 trillion US dollars in 2023, making it the second largest economy in the world after the United States.",
    ),

    # ── DEFINITIONS / CONCEPTS ───────────────────────────────────────────────
    (
        "What is a hedge fund?",
        "A hedge fund is a pooled investment fund that uses advanced strategies including leveraging, short-selling, and derivatives to generate returns for accredited investors. They are regulated less strictly than mutual funds.",
    ),
    (
        "What is the difference between machine learning and deep learning?",
        "Machine learning is a broad field where algorithms learn patterns from data. Deep learning is a subset of machine learning that uses artificial neural networks with many layers, popularised by Geoffrey Hinton, Yann LeCun, and Yoshua Bengio.",
    ),
]


# ── run ────────────────────────────────────────────────────────────────────────
def main() -> None:
    lines: list[str] = []
    lines.append("HyDE Distillation Evaluation — 50 queries")
    lines.append("=" * 72)
    lines.append("")

    for i, (query, hyde) in enumerate(CASES, 1):
        distilled = _distill_search_query(query, hyde)
        changed = distilled != query
        tag = "DISTILLED" if changed else "UNCHANGED"
        lines.append(f"[{i:02d}] {tag}")
        lines.append(f"     Q:       {query}")
        lines.append(f"     HyDE:    {hyde[:120]}{'...' if len(hyde) > 120 else ''}")
        lines.append(f"     SEARCH:  {distilled}")
        lines.append("")

    output_path = Path(__file__).resolve().parent.parent / "data" / "diagnostics" / "hyde_distill_eval.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(CASES)} results → {output_path}")
    print()
    # Also print to stdout for quick review
    print("\n".join(lines))


if __name__ == "__main__":
    main()
