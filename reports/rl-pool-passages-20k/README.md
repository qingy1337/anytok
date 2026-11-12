# Exact text behind the RL transfer gap

The largest contributors are passages of repeated image-link markup in the medical corpus. The selected examples also include Chinese blog-interface text, Japanese prose and a decimal-number copying passage.

For tokenizer 0635, the average gap rises from **+0.06637** on the first sample to **+0.10073 BPB** on the separate confirmation sample. The medical-domain change accounts for **75.5%** of that increase.

The largest confirmation contributor is **PMC10211047**: canonical **0.132 BPB**, transferred **3.573 BPB**, gap **+3.441 BPB**. It has 43 scored bytes and 32 scored canonical tokens.

The +0.101 number is an average across the complete evaluation sample. An individual passage can have a much larger gap. Each passage contributes according to its scored bytes within its domain, and each of the nine domains has equal weight.

## What changed between the two text samples

| Domain | First-sample gap | Confirmation gap | Contribution to average-gap increase |
|---|---:|---:|---:|
| medical-en | +0.16981 | +0.40323 | +0.02594 |
| math-en | -0.00574 | +0.06614 | +0.00799 |
| web-zh | +0.15744 | +0.18147 | +0.00267 |
| web-ar | +0.08589 | +0.09678 | +0.00121 |
| web-en | -0.02676 | -0.01807 | +0.00097 |
| web-ja | +0.12052 | +0.12621 | +0.00063 |
| web-hi | +0.03463 | +0.03136 | -0.00036 |
| legal-en | +0.02033 | +0.01012 | -0.00113 |
| code | +0.04119 | +0.00933 | -0.00354 |

## Exact passages

The prefix is the text fed through the transferred input embeddings. The continuation is the exact next 32 canonical tokens that were scored. Cuts can fall inside a word or filename. Line breaks and spelling below come from the stored evaluation data. The JSON retains exact whitespace and native prefix token IDs.

Examples are ordered by contribution to tokenizer 0635’s total gap, with confirmation examples first. We selected the ten largest contributors from each sample and the largest Chinese, Japanese and math contributors. This is a post-hoc inspection of saved losses; it does not establish a causal explanation.

### confirm · medical-en · PMC10211047

Canonical 0.1316 BPB; transferred 3.5731; gap +3.4415. Contribution to overall mean gap: +0.004602 BPB.

Prefix:

````text
pmc

![](warmedparis138453-0135){#sp1 .1105}

![](warmedparis13
````

Scored continuation:

````text
8453-0136){#sp2 .1106}

![](warmedparis1384
````

### confirm · medical-en · PMC10210727

Canonical 0.2755 BPB; transferred 3.6104; gap +3.3349. Contribution to overall mean gap: +0.004459 BPB.

Prefix:

````text
pmc

![](warmedparis138449-0107){#sp1 .250}

![](warmedparis13
````

Scored continuation:

````text
8449-0108){#sp2 .251}

![](warmedparis13844
````

### confirm · medical-en · PMC10210657

Canonical 0.1880 BPB; transferred 3.4995; gap +3.3115. Contribution to overall mean gap: +0.004428 BPB.

Prefix:

````text
pmc

![](warmedparis138450-0074){#sp1 .366}

![](warmedparis13
````

Scored continuation:

````text
8450-0075){#sp2 .367}

![](warmedparis13845
````

### confirm · medical-en · PMC10211621

Canonical 0.1859 BPB; transferred 3.6168; gap +3.4309. Contribution to overall mean gap: +0.004374 BPB.

Prefix:

````text
pmc

![](medexchic131447-0009){#sp1 .393}

![](medexchic131
````

Scored continuation:

````text
447-0010){#sp2 .394}

![](medexchic131447
````

### confirm · medical-en · PMC10211224

Canonical 0.0278 BPB; transferred 3.1926; gap +3.1649. Contribution to overall mean gap: +0.004232 BPB.

Prefix:

````text
pmc

![](warmedparis138454-0264){#sp1 .1514}

![](warmedparis138454-0265){#sp2 .1515
````

Scored continuation:

````text
}

![](warmedparis138454-0266){#sp3 .1516}

````

### confirm · medical-en · PMC10211756

Canonical 0.1525 BPB; transferred 3.3276; gap +3.1751. Contribution to overall mean gap: +0.004048 BPB.

Prefix:

````text
pmc

![](medexchic131460-0021){#sp1 .137}

![](medexchic131
````

Scored continuation:

````text
460-0022){#sp2 .138}

![](medexchic131460
````

### confirm · medical-en · PMC10211880

Canonical 0.0339 BPB; transferred 3.1100; gap +3.0760. Contribution to overall mean gap: +0.004018 BPB.

Prefix:

````text
pmc

![](medexchic131444-0005){#sp1 .317}

![](medexchic131444-0006){#sp2 .318}
````

Scored continuation:

````text


![](medexchic131444-0007){#sp3 .319}

![
````

### confirm · web-zh · <urn:uuid:cd63e180-5b51-461c-bbda-36b6890b5d2a>

Canonical 0.8920 BPB; transferred 2.1725; gap +1.2804. Contribution to overall mean gap: +0.004006 BPB.

Prefix:

````text
一個人的精彩 發表在 痞客邦
````

Scored continuation:

````text
 PIXNET 留言(0) 人氣()
愉快的暑假快
````

### confirm · medical-en · PMC10211525

Canonical 0.1899 BPB; transferred 3.2322; gap +3.0422. Contribution to overall mean gap: +0.003879 BPB.

Prefix:

````text
pmc

![](medexchic131434-0018){#sp1 .70}

![](medexchic13
````

Scored continuation:

````text
1434-0019){#sp2 .71}

![](medexchic131434
````

### confirm · medical-en · PMC10211918

Canonical 0.0378 BPB; transferred 2.9635; gap +2.9257. Contribution to overall mean gap: +0.003821 BPB.

Prefix:

````text
pmc

![](medexchic131463-0020){#sp1 .226}

![](medexchic131463-0021){#sp2 .227}
````

Scored continuation:

````text


![](medexchic131463-0022){#sp3 .228}

![
````

### confirm · web-ja · <urn:uuid:60eed7ec-e74a-4813-9371-ec6836c5cb39>

Canonical 1.4870 BPB; transferred 1.9905; gap +0.5035. Contribution to overall mean gap: +0.001497 BPB.

Prefix:

````text
コンペ用の通称「痛アッガイ」、中々前に進みません。
塗ってはヤスリ、塗
````

Scored continuation:

````text
ってはヤスリの無間地獄は何とか
````

### confirm · math-en · https://convertoctopus.com/2-8-months-to-weeks

Canonical 0.0008 BPB; transferred 0.7800; gap +0.7791. Contribution to overall mean gap: +0.001460 BPB.

Prefix:

````text
475 weeks

## Alternative conversion

We can also convert by utilizing the inverse value of the conversion factor. In this case 1 week is equal to 0.082137210209655 × 2.8 months.

Another way is
````

Scored continuation:

````text
 saying that 2.8 months is equal to 1 ÷ 0.082137210209655
````

### screen · medical-en · PMC10210884

Canonical 0.1780 BPB; transferred 3.6720; gap +3.4939. Contribution to overall mean gap: +0.004010 BPB.

Prefix:

````text
pmc

![](warmedparis138452-0196){#sp1 .902}

![](warmedparis13
````

Scored continuation:

````text
8452-0197){#sp2 .903}

![](warmedparis13845
````

### screen · medical-en · PMC10211625

Canonical 0.1988 BPB; transferred 3.3088; gap +3.1100. Contribution to overall mean gap: +0.003569 BPB.

Prefix:

````text
}

![](medexchic131447-0016){#sp5 .400}

![](medexchic131447-0017){#sp6 .401}

![](me
````

Scored continuation:

````text
dexchic131447-0018){#sp7 .402}

![](medexch
````

### screen · medical-en · PMC10211691

Canonical 0.1610 BPB; transferred 3.4145; gap +3.2535. Contribution to overall mean gap: +0.003560 BPB.

Prefix:

````text
pmc

![](medexchic131450-0005){#sp1 .461}

![](medexchic131
````

Scored continuation:

````text
450-0006){#sp2 .462}

![](medexchic131450
````

### screen · medical-en · PMC10210798

Canonical 0.1893 BPB; transferred 3.2727; gap +3.0834. Contribution to overall mean gap: +0.003539 BPB.

Prefix:

````text
pmc

![](warmedparis138451-0072){#sp1 .626}

![](warmedparis13
````

Scored continuation:

````text
8451-0073){#sp2 .627}

![](warmedparis13845
````

### screen · medical-en · PMC10211657

Canonical 0.1685 BPB; transferred 3.3691; gap +3.2006. Contribution to overall mean gap: +0.003502 BPB.

Prefix:

````text
pmc

![](medexchic131461-0013){#sp1 .159}

![](medexchic131
````

Scored continuation:

````text
461-0014){#sp2 .160}

![](medexchic131461
````

### screen · medical-en · PMC10211889

Canonical 0.0252 BPB; transferred 2.9187; gap +2.8935. Contribution to overall mean gap: +0.003166 BPB.

Prefix:

````text
exchic155363-0013){#sp1 .339}

![](medexchic155363-0014){#sp2 .340}

![](medexchic155
````

Scored continuation:

````text
363-0015){#sp3 .341}

![](medexchic155363
````

### screen · medical-en · PMC10211728

Canonical 0.0457 BPB; transferred 2.5674; gap +2.5217. Contribution to overall mean gap: +0.002894 BPB.

Prefix:

````text
pmc

![](medexchic131456-0010){#sp1 .6}

![](medexchic131456-0011){#sp2 .7
````

Scored continuation:

````text
}

![](medexchic131456-0012){#sp3 .8}

![](
````

### screen · web-ja · <urn:uuid:c135b73d-0aba-4a60-8da6-a9308d209a7e>

Canonical 0.5114 BPB; transferred 1.0822; gap +0.5708. Contribution to overall mean gap: +0.001802 BPB.

Prefix:

````text
解決する瞬間まで待つべきなのでしょうか？」
「それは苦痛を伴う出来
````

Scored continuation:

````text
事なのでしょうか？」
「それを経験
````

### screen · web-zh · <urn:uuid:4d35b0eb-237f-41de-b3a5-737e9fc3e7e4>

Canonical 1.2771 BPB; transferred 1.8151; gap +0.5380. Contribution to overall mean gap: +0.001788 BPB.

Prefix:

````text
光、动感是青春广播对频率外在形象的整体要求。而敏锐、包容、学习、激
````

Scored continuation:

````text
情则是青春调频对频率内容风格的整
````

### screen · web-zh · <urn:uuid:2239b328-9a92-4e7f-84ff-909ba92aaeec>

Canonical 1.4750 BPB; transferred 1.9502; gap +0.4752. Contribution to overall mean gap: +0.001678 BPB.

Prefix:

````text
了不知道幾百遍，直到今年才投入經營民宿。其實，目前為止，一切都十分順
````

Scored continuation:

````text
利，民宿的施工品質，建材，都是理想
````

### screen · math-en · https://www.jiskha.com/display.cgi?id=1323279198

Canonical 0.1169 BPB; transferred 0.6371; gap +0.5202. Contribution to overall mean gap: +0.001486 BPB.

Prefix:

````text
asses through the basketball hoop mounted at 3.05 …
3. ### physics

A basketball player is standing on the floor 10.0 m from the basket as in the figure below. The height of the basket is H = 3.05 m, and he shoots the ball at
````

Scored continuation:

````text
 an angle, θ = 37.6°, with the horizontal from a height of h = 2.20 …
4. ### physics
````

## Data and reproduction

* [passages.json](passages.json): the selected exact text, native prefix tokenization, and all 576 passage-level losses/contributions for both tokenizers.
* [analysis.json](analysis.json): domain decomposition and the leading passage.
* [manifest.json](manifest.json): file sizes and SHA-256 hashes.

The source panel stays in `blackpantherx86 / anytok-rl-all-pool-20k / panel.json`. The cloud extraction validated the complete panel hashes, ordered document IDs, canonical/native tokenizer file hashes, prefix round trips, and 32-token continuation byte counts. Only selected short passages were retrieved locally to answer the request.

Recreate this report with `python scripts/analyze_rl_pool_passages.py --input reports/rl-pool-passages-20k/passages.json`.
