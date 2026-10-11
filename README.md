# Ryan Kunkel Johns Hopkins CS 745: Reasoning Under Uncertainty

This repo serves to house and manage my implementation for Johns Hopkins EN.605.745 Reasoning under uncertainty.

My project aims to accomplish an implementation that supports my "Predictive Specification via Entropy Reduction" research direction.
At a high level, we want to accomplish the following

- Take an existing specification for a system, defined by formal requirements
- Parse out the ontology of the system in terms of entities, and the relationships between them
- Form a knowledge graph of the entities and their relationships
- Measure the entropy of the relationships established in the KG

This will allow us to determine if we have specified our system well in terms of requiremnents, and give us a unique angle to determine how much risk is inherent in our system.

Also, we are going to implement in rust as much in possible, which might leave you asking:  
"why"  
because we can, thats why.

This tool will help engineers minimize design risk by encouraging specifications that are consistent, complete, noncontradictory, and unambiguous. We introduce a metric, shannon entropy, measured in nats, to quantify the uncertainty in the probability distribution over possible interpretations of a specification. This is calculated as $(H=-\sum_i p_i\ln(p_i))$. Higher entropy indicates greater ambiguity across interpretations, while lower entropy indicates a more concentrated distribution.

We are accepting that natural langauge in inherently uncertain, and we are going to deal with that directly: We will measure the uncertainty, and reason about what needs to be specified to reduce it. This is the core idea that makes the project applicable to the course.

This project rests on two core supporting reasoning frameworks / modeling architectures:

- GLiNER
  Generalizable lightweight named entity recognition. This is how we are going to build structure from our unstructed natural langauge requirements. We are going to design a schema that will allow the NER pipeline to establish the triplets will use for training later

- rotatE  
  rotatE is a knowledge graph embedding algorithm that will establish asymetric relationships by embedding the entities as complex valued vectors. We use the results from GLiNER and run them through the rotate algorithm, which embeds the entities in the complex plane, and encodes the relationships as a complex exponential that will rotate the entity in the complex plane (hence the name rotatE). The algorithm uses a self supervised constrastive learning approach which is said to have "learned" the triplet set if the embeddings are placed in the complex plane, such that applying the relationship vector to an entity rotates that vector on top of the entity it relates to.

## How to produce the results for yourself

- Clone the repo locally, and cd into it.

- Make sure you have the hugging face cli ready to go on your system:  
  `uv tool install huggingface-hub`

- Download the gliner2.5 base model (or a different gliner model if youre interested) by running this command in the terminal:  
  `hf download fastino/gliner2.5-base-v1 --local-dir ./models/gliner2.5-base`

- Run the project, passing in the training flag:
  `cargo run --release -- --train`
  _here the training is performing KG construction using rotatE_

- Test a query
  `cargo run --release -- --query "head relation tail"`

  ex:
  `cargo run --release -- --query "tas communicates_via ?"`

  Make sure you use the release flag, it will dramatically speed up training as rustc will optimize the binary

### Code Sources

I am relying on the rotatE source from this repo: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding.  
instead of asking you to clone this repo, i committed a thin copy with just the python scripts and point to them in config.json for convienence.

I had to adopt a variation of the GLiNER inference port for rust in this repo:
https://github.com/apiplant/gliner-rs

This repo had some issues when trying to run my tests locally, so i had to fork this repo and bugfix the batch processing logic, to deduplicate chuncks that appear in the batch overlap, meaning I am using my fork here: https://github.com/rykunk21/gliner-rs

## Initial Results

At the rough draft phase, I am quite happy with the initial results.

in our requirements text, I have two subsystems: Command and Control interface (C2I) and Threat Assessment Subsystem (TAS).

### Test 1:

In our specification we have the requirement

```
If the system is networked, then the C2I shall communicate using VMF.
```

we do not have a similar requirement for TAS. We perform two queries for C2I and TAS with the communicates_via relationship

#### Query: `c2i communicates_via ?`

`Entropy: 0.0712`

| Rank |   Prob |   Dist | Candidate               |
| ---: | -----: | -----: | ----------------------- |
|    1 | 0.9929 |  2.485 | vmf                     |
|    2 | 0.0003 | 10.477 | vmf_k-series_messages   |
|    3 | 0.0003 | 10.566 | operator_confirmation   |
|    4 | 0.0003 | 10.732 | safety_interlock_module |
|    5 | 0.0003 | 10.750 | threshold               |
|    6 | 0.0002 | 10.836 | operator                |
|    7 | 0.0002 | 10.856 | eo/ir_sensor            |
|    8 | 0.0002 | 10.955 | networked               |
|    9 | 0.0002 | 10.965 | tem₀₀_mode              |
|   10 | 0.0002 | 11.018 | communications_module   |

Here, we feel confident that c2i communicates via the variable message format (VMF). We can see almost all of the probability is concentrated on the correct tail given this head.

now looks what happens when we ask about TAS.

#### Query: `tas communicates_via ?`

`Entropy: 4.0030`

| Rank |   Prob |   Dist | Candidate                |
| ---: | -----: | -----: | ------------------------ |
|    1 | 0.1196 |  9.150 | pd                       |
|    2 | 0.0384 | 10.284 | airspace                 |
|    3 | 0.0346 | 10.391 | friendly_force_positions |
|    4 | 0.0329 | 10.440 | false_alarms             |
|    5 | 0.0326 | 10.450 | tem₀₀_mode               |
|    6 | 0.0296 | 10.545 | visible_imagery          |
|    7 | 0.0283 | 10.591 | response_module          |
|    8 | 0.0276 | 10.617 | ku-band                  |
|    9 | 0.0259 | 10.679 | communications_module    |
|   10 | 0.0248 | 10.721 | classification_decision  |

we do not really know how TAS communicates, as it is not specified in requirements. The distribution is almost uniform across all possible tails, which results in high entropy

_now look what happens after we add this requirement to our reqts.txt, and change nothing else about our training parameters or configuration_

```
If the system is networked, then the TAS shall communicate using VMF.
```

#### Query: `tas communicates_via ?`

`Entropy: 0.4019`

| Rank |   Prob |   Dist | Candidate                     |
| ---: | -----: | -----: | ----------------------------- |
|    1 | 0.9511 |  4.370 | vmf                           |
|    2 | 0.0020 | 10.531 | beam_formation_module         |
|    3 | 0.0016 | 10.740 | composite_air_picture_updates |
|    4 | 0.0016 | 10.789 | specified_limits              |
|    5 | 0.0015 | 10.855 | threshold                     |
|    6 | 0.0014 | 10.881 | airborne_object               |
|    7 | 0.0013 | 10.936 | 3d_path                       |
|    8 | 0.0013 | 10.968 | tas                           |
|    9 | 0.0013 | 10.969 | overlapping_engagements       |
|   10 | 0.0013 | 10.992 | host_vehicle                  |

The distribution shifts immediately to support the correct answer, and the entropy drops across the distribution by a factor of 10

To me, this is just incredible. We are extracting knowledge from unstructured text by projecting entities into the complex plane and rotating them.

### Test 2

Now we want to ensure that we are capturing contradictions. Lets add two contradictory requirements to our requirement set

```
If the system is networked, then the TAS shall communicate using VMF.
if the system is networked, then the TAS shall communicate using the radar.
```

And train / query again:

#### Query: `tas communicates_via ?`

`Entropy: 0.7903`
| Rank | Prob | Dist | Candidate |
|-----:|-------:|-------:|--------------------------------------|
| 1 | 0.5718 | 4.209 | radar |
| 2 | 0.4049 | 4.554 | vmf |
| 3 | 0.0012 | 10.371 | 3d_path |
| 4 | 0.0010 | 10.538 | sensor |
| 5 | 0.0010 | 10.544 | state_awaiting_operator_confirmation |
| 6 | 0.0010 | 10.579 | closest_point_of_approach |
| 7 | 0.0007 | 10.888 | kinetic_destruction |
| 8 | 0.0007 | 10.973 | adjacent_systems |
| 9 | 0.0006 | 11.079 | response_module |
| 10 | 0.0006 | 11.135 | confidence_metric |

We can see that now specifying contradictory requirements also spreads the distribution across both links, increasing entropy. This shows that our entropy metric accounts for contradiction.

## Thoughts on initial results

When coding the initial draft up, I make some discoveries I wanted to reflect on at this phase.

These initial results look promising, but the stregnth of the results may be due to overfitting. I say this because testing on non represented requirements results in almost always a pure uniform distribution, while represented requirements are near zero entropy.

I still believe the general idea of entropy reduction is corect. However, we want to avoid reducing entropy just by just memorizing the training data used for rotatE. This was a problem I didnt see coming when initially devised this metric and methodology for the approach.

_I plan on dealing with this in two ways:_

- We should still being doing validation / test partitions, measuring Mean Rank, Mean Reciprocal Rank, and Hits@n, as done in the rotatE paper. This will give us a general idea to how well we are generalizing to our held out test set. This is a traditional machine learning appraoch to detect and report on overfitting. Nothing crazy here.

- I figured it would also be smart to include the total entropy of all relationship combinations. This might seem counterintuitive, but hear me out: we want to avoid specifying every implicit requirement explicitly which is what motivated the use of rotatE for link prediction in the first place. The problem is that a new entity brings a huge entropy cost, since it creates new distributions against every existing entity. We deal with this by leaning on the hierarchical nature of requirements. Requirements get expressed at a high level and decomposed into more particular ones, with entropy only contributing to its parent. This dramatically cuts the connections a new entity introduces, and also helps automated architecture: when runaway entropy shows up again, we decompose into another child layer.

## Continuation plan

We still need to implement the baseline 'naive' test that shows how this better than just giving the requirement set to an LLM and measuring the probability distribution over the continuation tokens. I expect this to perform worse, as the context tokens will recover the hallucination problem.

I also need to generate a larger dataset for my problem. Right now I only have around 50 requirements. It would be useful to have a much larger dataset and still demonstrate that I can represent a more densely populated complex embedding space.

I also need to include partitioning the layers to only include entropy contribtions from the child tokens. This is a structural requirements setup, where a top level requirement. Given this, i will be able to have well considered data and metrics that support my final write up of the project in the paper.

I will continue to enahnce the codebase to add more resiliant error handling and logging with the tracer library. One consideration will be adding an indexing layer to the entities. I would like to use an embedding strategy to deduplicate requirements that have the same meaning. vmf and vmf-k-series-messages are treated independently. We might be able to achieve this with no indexing layer by investigating an ehanced schema design for GLiNER to use.

Time permitting, I will also try to rewrite the rotatE algorithm in rust, so I can organize this project as a rust library i can use for later projects.
