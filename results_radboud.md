Evaluation of models trained on the NB-A-43 Radboud train set.

# Ablations: Shape, Texture, and Arrangement

## Shape-based Models

### Embedding: NB-A-51

| PE | Attention Pattern | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard RoPE | k-NN, k=24 | **0.8588** | 0.7917 | 0.8597 | **0.8582** | **0.8933**<br />[0.883, 0.9024] | **0.9351**<br />[0.9308, 0.9392] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/c5fe3bbb1c4e4f18916a87ae72df180f) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ca383cf16b414711b10b8ace1980e066), report |
| None | k-NN, k=24 | 0.8117 | 0.7205 | 0.8353 | 0.797 | 0.8392<br />[0.8276, 0.8497] | 0.8981<br />[0.8923, 0.9036] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/f2812cd2231747798c0b0b67f72f994b) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/c56d3d81f75b43f2b14d7e11e7393ea4), report |
| None | dense | 0.7699 | 0.6643 | 0.814 | 0.7423 | 0.787 [0.7728, 0.7998] | 0.8616 [0.8536, 0.8687] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/96b2096d540f4abd980be77f40733d81) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ee8253d41da84ff3bce1f155ce7a6885), [report](https://xopat.rationai.cloud.trusted.e-infra.cz/index.php?p=public_mlflow%2F37%2Fa9f012648cfb4aa5b5696a51d424c270%2Fartifacts%2Freport&view=report.html) |
| standard RoPE | dense | 0.8359 | 0.7561 | 0.8472 | 0.8288 | 0.8689 [0.8574, 0.8797] | 0.9176 [0.9122, 0.923] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/9783cd5bbaca46cf9a0dbca97f45073a) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/dfa9568715424f248dd9f629bbeb4744), [report](https://xopat.rationai.cloud.trusted.e-infra.cz/index.php?p=public_mlflow%2F37%2Fae43ea87d748446b9dabffd908daabcc%2Fartifacts%2Freport&view=report.html) |

#### "k" Sweep

* with standard RoPE

| k | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.8566 | 0.7897 | 0.8555 | 0.8573 | 0.8916<br />[0.8815, 0.9007] | 0.9332<br />[0.9290, 0.9374] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/3392b78137ee4d11914ee76b886f67ff) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/1f3b5542c3f4496294f2bbcfb6081ae8) |
| 12 | 0.8564 | 0.7872 | 0.8596 | 0.8544 | 0.8907<br />[0.8807, 0.9000] | 0.9334<br />[0.9293, 0.9375] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/74bb7b7f11334e88b3be7289128f8cc8) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/6e6f2756dd284ccb9a88864b30ad27fe) |
| 16 | 0.8497 | 0.7789 | 0.8515 | 0.8486 | 0.8829<br />[0.8727, 0.8923] | 0.928<br />[0.9233, 0.9322] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/0ef3b54920e3428a9933dae1e0eeeac9) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/843348978ac5456bba850b8e73108742) |
| 20 | 0.857 | 0.7872 | 0.8615 | 0.8541 | 0.8916<br />[0.8812, 0.9008] | 0.9340<br />[0.9294, 0.9382] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/e378b2b99a60496caddecc92e2e31cb9) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/9a7a562a7e0148d496e34aabd8303a8e) |
| <span style="background-color:palegreen;">**24**</span> | **0.8588** | **0.7917** | 0.8597 | **0.8582** | **0.8933**<br />[0.883, 0.9024] | **0.9351**<br />[0.9308, 0.9392] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/c5fe3bbb1c4e4f18916a87ae72df180f) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ca383cf16b414711b10b8ace1980e066) |
| 32 | 0.8535 | 0.7878 | 0.8481 | 0.8569 | 0.8878<br />[0.8771, 0.8971] | 0.9304<br />[0.9257, 0.9349] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/6b29452a67654c0385ccc02b15999406) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/651856fca27c46ca92d9c6b8c64fd02a) |
| 40 | 0.8572 | 0.7874 | **0.8619** | 0.8542 | 0.8916<br />[0.8818, 0.9009] | 0.9344<br />[0.9301, 0.9384] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/8e4834b363d743b4a089d8004eff7f5e) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/fca91fc11ab74180bdb138a5812c8400) |

## Texture-based Models

### Embedding: NB-A-52

| PE | Attention Pattern | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard RoPE | k-NN, k=16 | 0.9022 | 0.8449 | 0.9139 | 0.8949 | 0.9377 [0.9304, 0.9443] | 0.9653 [0.9626, 0.9681] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/843f60f505c84174861d34e1948cece1) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/0946c406d23249ccaf5baab638cddab0) |
| None | k-NN, k=16 | 0.8935 | 0.8313 | 0.9076 | 0.8846 | 0.9299 [0.9225, 0.9364] | 0.9604 [0.9574, 0.9633] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/325eb62111b94ab7ae9d08fb72bad7c8) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/716fbf2b243e4ce999253bdbc1345c8d) |
| standard RoPE | dense | 0.7535 | 0.63 | 0.8723 | 0.6790 | 0.7485<br />[0.7270, 0.7686] | 0.8515<br />[0.8433, 0.8597] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/45d1cd7e7f3b4678acd0efd761e6db39) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ef3ca76687194e9c9f35ae48cdc2ab64) |
| None | dense | 0.7144 | 0.5777 | 0.9607 | 0.7633 | 0.7633<br />[0.7435, 0.7805] | 0.8652<br />[0.8569, 0.8733] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/0fa45b5652e4411199394c7d6677479d) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/770343c7068744838a580c809b5a391b) |

## Arrangement-based Models

### Embedding: NB-A-103

| PE | Attention Pattern | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| "V" rotation RoPE | k-NN, k=32 | **0.7463** | **0.6318** | **0.8180** | **0.7014** | **0.7403**<br />[0.7216, 0.756] | **0.8377**<br />[0.8288, 0.8457] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/a07f92db18494dd1a58c54f5736098b9) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/e2a24fb86854407f96d5cf6295259fa0) |
| "V" rotation RoPE | dense | 0.6727 | 0.5508 | 0.8144 | 0.5839 | 0.6673<br />[0.6453, 0.6874] | 0.78<br />[0.7667, 0.7917] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/7ee624ff55cb4e269a7c0eaaed75ca17) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/9a6e7913e39c49d8988d2c3d9b3025ed) |

#### "k" Sweep

* with standard RoPE

| k | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.7391 | 0.6212 | 0.8266 | 0.6843 | 0.7396<br />[0.7222, 0.7554] | 0.8355<br />[0.8274, 0.8434] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/7e99a96e56074803a249cfda67e040b9) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/a130cd1881aa4c50959367672079b5bc) |
| 12 | 0.7372 | 0.62 | 0.8209 | 0.6848 | 0.7252<br />[0.7044, 0.7434] | 0.83<br />[0.8218, 0.8388] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/c9f33ed180354a42ae4eb4ccbc63e8d9) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/6e26ee757f244aa0a5a5db952ba78722) |
| 16 | 0.7385 | 0.6198 | 0.8302 | 0.6811 | 0.7342<br />[0.7174, 0.7498] | 0.8348<br />[0.8256, 0.8431] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/d3c4d6932f8c45e796d321537c1f1d5d) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/578e096cef3d46488237da79a0a849f0) |
| 20 | 0.7349 | 0.6162 | 0.8267 | 0.6775 | 0.7228<br />[0.7026, 0.7408] | 0.8293<br />[0.8201, 0.838] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/685b3830baf6468baada5b76c3d8b576) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/4f13e7fdb2684927bf3f115c5ff4675b) |
| 24 | 0.7403 | 0.6212 | **0.835** | 0.6811 | 0.7360<br />[0.7181, 0.7526] | 0.8365<br />[0.8279, 0.8447] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/fed2af54e23f4e84909769c3d5a7e16e) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/5bb034739794495790607e25aa751840) |
| <span style="background-color:palegreen;">**32**</span> | **0.7463** | **0.6318** | 0.8180 | **0.7014** | **0.7403**<br />[0.7216, 0.756] | **0.8377**<br />[0.8288, 0.8457] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/a07f92db18494dd1a58c54f5736098b9) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/e2a24fb86854407f96d5cf6295259fa0) |
| 40 | 0.7363 | 0.619 | 0.8202 | 0.6837 | 0.7299<br />[0.7119, 0.7466] | 0.8306<br />[0.8212, 0.839] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/4ca5ec36ed1c478f8112cd49c11ab0fd) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/6117dcd27fdd43a8b8e926633d15ce36) |

### Embedding: NB-A-89

| PE | Attention Pattern | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard RoPE | k-NN, k=20 | **0.7758** | **0.6662** | **0.8372** | **0.7373** | **0.7871**<br />[0.7721, 0.8006] | **0.8683**<br />[0.861, 0.8752] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/b47c548043bb4781a4220bb307dde914) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/d4396cd0c7d442378b6c1ae16d9c13fb) |
| None | k-NN, k=20 | 0.7142 | 0.5940 | 0.8150 | 0.6510 | 0.7085<br />[0.689, 0.7266] | 0.8102<br />[0.8001, 0.8199] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/0039054daba24dbab683955bbb6cf275) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/c8b46fcb3bf840169f214c325a4a6d9d) |
| standard RoPE | dense | 0.7489 | 0.6367 | 0.8104 | 0.7103 | 0.7532<br />[0.7368, 0.7686] | 0.8414<br />[0.8324, 0.8501] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ce733803b39446419f91428ea6063ba4) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/342111f4912d465abeab920e03e55309) |
| None | dense | 0.6385 | 0.5194 | 0.8229 | 0.523 | 0.6274<br />[0.6068, 0.6477] | 0.7473<br />[0.7336, 0.7608] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/979c340e92ce42b9abe5f0a9a328d930) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/d010a080c18d4f1e9ecf4af0744f135f) |

#### "k" Sweep

* with standard RoPE

| k | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.7731 | 0.6625 | 0.8375 | 0.7327 | 0.785<br />[0.7702, 0.7989] | 0.8666<br />[0.8593, 0.8731] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/3d3aef64d73044d9b1714b4a7bfc04ea) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/af8766b73090424880be8cb9911e036c) |
| 12 | 0.7754 | **0.666** | 0.8363 | **0.7373** | 0.7855<br />[0.7713, 0.7993] | 0.8677<br />[0.8602, 0.8744] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ae3bdbaaed594dcdbe07d4b2367c7f08) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/23d7e8cf95e1441481b46f50f3bc3abd) |
| 16 | 0.7719 | 0.6603 | 0.8398 | 0.7294 | 0.7836<br />[0.7685, 0.7973] | 0.8659<br />[0.8590, 0.873] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/180feadf33a14b0d9abdd62d8665f5bb) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/da9d7f98bde146f2a4f4e12de29c6885) |
| <span style="background-color:palegreen;">**20**</span> | **0.7758** | 0.6662 | 0.8372 | **0.7373** | **0.7871**<br />[0.7721, 0.8006] | **0.8683**<br />[0.861, 0.8752] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/b47c548043bb4781a4220bb307dde914) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/d4396cd0c7d442378b6c1ae16d9c13fb) |
| 24 | 0.7745 | 0.6631 | **0.8426** | 0.7318 | 0.7854<br />[0.7704, 0.7997] | 0.8680<br />[0.8608, 0.8751] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/fced6298381a4c2b97e99b0f63e3adc1) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/5e011433ca31472c9e6311fdbbc4626b) |
| 32 | 0.7746 | 0.6643 | 0.8385 | 0.7345 | 0.7858<br />[0.7699, 0.7994] | 0.8673<br />[0.8603, 0.8744] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/d89853331e7d4688b24b05ae9c8307b3) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/e565e53c855f4575a7d22faa340ca73f) |
| 40 | 0.7739 | 0.6624 | 0.8422 | 0.7311 | 0.7867<br />[0.7729, 0.8001] | 0.8681<br />[0.8611, 0.8750] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/351c02f5e7824549b8aba91fabfe69e4) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/3be1c2e8156140ab9df23b42a36b0fc6) |

## Shape + Arrangement-based Models

### **Embedding**: NB-A-51 + [Relative-Position Value Attention](https://youtrack.rationai.cloud.e-infra.cz/articles/NB-A-106/Relative-Position-Value-Attention)

| PE | Attention Pattern | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard RoPE | k-NN, k=40 | **0.8728** | 0.8101 | **0.8749** | 0.8715 | **0.9083**<br />[0.8979, 0.9173] | **0.9454**<br />[0.9414, 0.9491] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/cdb3ba736846434aa75eb6396b7af9db) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/b3fc5aeedd8046f39c370fb76eab0631) |

#### "k" Sweep

* with standard RoPE

| k | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 |  |  |  |  |  |  | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/545b4ae6840c49d48e8182d917cad3cd) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/afd50a91829645b1889fde38e4b7a884) |
| 12 | 0.8612 | 0.7993 | 0.8542 | 0.8657 | 0.8971<br />[0.8875, 0.9063] | 0.9368<br />[0.933, 0.9404] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/f4d4732210f147e5ba94120a36316b3a) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/9b66cb04b46149cc91f14c38d0f58c52) |
| 16 | 0.8649 | 0.8020 | 0.8620 | 0.8667 | 0.8999<br />[0.8893, 0.9094] | 0.9397<br />[0.9358, 0.9434] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/9b5a129b90f742e1b9a42f5806556818) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ff5b792c496a4b1699b996fc0ddfcc6f) |
| 20 | 0.869 | 0.8087 | 0.8642 | 0.872 | 0.9040<br />[0.8936, 0.9131] | 0.9422<br />[0.9383, 0.9458] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/ba215df58d8e472e83b9df23a67f9051) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/899146cd094c4899ae82f67c7eacc7c7) |
| 24 | 0.8704 | 0.8096 | 0.8679 | 0.8721 | 0.9065<br />[0.8963, 0.9151] | 0.9439<br />[0.9402, 0.9476] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/090ac786bd89498bac2271e0394a56e0) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/4dda120e5f154736866cb6c7d8cbf5b6) |
| 32 | 0.8722 | **0.8104** | 0.8723 | **0.8721** | 0.9076<br />[0.8973, 0.917] | 0.9447<br />[0.9409, 0.9482] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/2517da0dcbbc40f3be96a580e4a60889) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/8d3859ca64674bd4b4407b8ac66a3a87) |
| <span style="background-color:palegreen;">**40**</span> | **0.8728** | 0.8101 | **0.8749** | 0.8715 | **0.9083**<br />[0.8979, 0.9173] | **0.9454**<br />[0.9414, 0.9491] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/cdb3ba736846434aa75eb6396b7af9db) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/b3fc5aeedd8046f39c370fb76eab0631) |

### Embedding: NB-A-51 + NB-A-89 (via concatenation)

| PE | Attention Pattern | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard RoPE | k-NN, k=32 | 0.8607 | 0.7957 | 0.8587 | 0.8619 | 0.8952<br />[0.8852, 0.9047] | 0.9365<br />[0.9323, 0.9409] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/c33773a8f4a24f1f89fd1b7e813535ab) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/59fcb8e40b204e1cbb6a152544ebbb18) |

#### "k" Sweep

* with standard RoPE

| k | Accuracy | Precision | Recall | Specificity | AUPRC [95% CI] | AUROC [95% CI] | Training | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.8586 | 0.7923 | 0.8577 | 0.8592 | 0.8931<br />[0.8827, 0.9025] | 0.935<br />[0.9309, 0.9389] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/a99a5f19d56144198c04a5fdb82773c8) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/964f312aab6148be81a50c586e2c5e06) |
| 12 | 0.8575 | 0.7894 | 0.8591 | 0.8564 | 0.8908<br />[0.8800, 0.9001] | 0.9334<br />[0.9291, 0.9376] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/9dab0131d53448149ae845c0bb408705) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/f914e04fc57c444fbaa35a99df7f76b0) |
| 16 | 0.8596 | 0.7951 | 0.8560 | 0.8618 | 0.8929<br />[0.8829, 0.9023] | 0.9348<br />[0.9305, 0.9391] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/d8bd72460aef495bb6fb2c190af97797) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/2f42177bf675439aa622b810ebf181c4) |
| 20 | 0.8583 | 0.7877 | **0.8656** | 0.8538 | 0.8923<br />[0.8821, 0.9022] | 0.9346<br />[0.9302, 0.939] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/0f35b8480b7645f48966cc366784c8e4) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/3d1d1537a5c64e868cef3b4cf65d2cca) |
| 24 | 0.8584 | 0.7895 | 0.8623 | 0.856 | 0.8937<br />[0.8833, 0.9030] | 0.9355<br />[0.9312, 0.9396] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/bf720ed1feb5411fa4f844c3a80600f7) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/60e1f17d9a7d4becaafb28e36593918a) |
| <span style="background-color:palegreen;">**32**</span> | **0.8607** | 0.7957 | 0.8587 | 0.8619 | **0.8952**<br />[0.8852, 0.9047] | **0.9365**<br />[0.9323, 0.9409] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/c33773a8f4a24f1f89fd1b7e813535ab) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/59fcb8e40b204e1cbb6a152544ebbb18) |
| 40 | 0.8600 | **0.7962** | 0.8556 | **0.8628** | 0.8948<br />[0.8841, 0.9040] | 0.9357<br />[0.9309, 0.94] | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/d34ce9ca18874320ad86d559a7e3649e) | [URI](https://mlflow.rationai.cloud.e-infra.cz/#/experiments/37/runs/53465befdaf9498fa6156be4397ffb78) |
