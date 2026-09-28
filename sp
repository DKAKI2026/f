Accroche (30 s)

« Chaque année, une compagnie d'assurance doit mettre de l'argent de côté pour des décès qui ne sont pas encore arrivés. Elle ne sait ni quand, ni pour qui. Mais elle doit savoir combien.

Trop peu : elle ne pourra pas tenir ses promesses aux familles. Trop : elle immobilise de l'argent qui aurait pu faire baisser les primes.

Problématique (45 s)
Notre objectif : prévoir les montants de prestations de décès. Nous ne partons pas de zéro. La table de mortalité est notre point de départ, pas notre adversaire. Elle est solide, mais elle décrit une moyenne, pas nos assurés.

La question devient : comment s'appuyer sur la table sans s'y enfermer, et sur nos propres données sans se laisser tromper par le hasard ?

Trois exigences. La crédibilité : savoir combien croire nos données. La fiabilité : prévoir juste sur ce qu'on n'a pas encore vu, y compris les gros montants. L'évolutivité : que la méthode tienne dans les années qui viennent.

Un modèle qui ne regarde que ses performances (score) ne répond pas à ces trois questions.

Diapo 3 · Notre réponse (20 s)

« Trois briques, au service d'une seule tâche : prévoir les montants. Un facteur qui dose la confiance en nos données, un modèle qui affine la prévision, une hypothèse de tendance pour les années à venir. »
 Diapo 4 
 Étape	Question	Réponse affichée
1. Le facteur	Que s'est-il vraiment passé ?	On compare le réel à ce que la table prévoyait. L'écart donne un coefficient qui ajuste la table.
2. Les strates	Pour qui ?	On le calcule par groupes d'assurés qui se ressemblent (sexe, fumeur, capital, âge).
3. La crédibilité	Peut-on s'y fier ?	Plus le groupe est grand, plus on suit ses données. S'il est petit, on reste près de la table.
4. Les autres critères	Est-ce bien réglé ?	D'autres méthodes reconnues refont le calcul et vérifient le résultat.

Nous partons de la table, que nous ajustons au moyen d'un facteur, coefficient tiré de l'écart entre les prestations observées et celles que la table prévoyait, établi séparément pour chaque strate, c'est-à-dire pour chaque groupe d'assurés aux caractéristiques semblables, dans la mesure où la crédibilité de ce groupe le justifie, et nous validons ce réglage au moyen d'autres critères reconnus dans l'industrie : la fluctuation limitée fondée sur l'intervalle de confiance, la formule de Whitney, la méthode de Bühlmann-Straub et l'approche bayésienne. »
Diapo 5 · Le moteur, LightGBM (50 s)
« Le facteur ne regarde que quelques critères à la fois. Le deuxième maillon, c'est un modèle d'apprentissage automatique, LightGBM. Il regarde tous les critères ensemble et apprend ce qu'il reste à corriger, en partant toujours du facteur. On ne repart donc jamais de zéro.
Diapo 6 ·  le carburant (50 s)

« Un moteur ne vaut que par ce qu'on lui donne. Nous avons enrichi les données avec des colonnes que les actuaires reconnaissent.
Nom dans le code	Ce que c'est	Pourquoi les actuaires s'y intéressent	Statut
AttdAge	L'âge de l'assuré au moment de l'observation, c'est-à-dire son âge à la souscription augmenté du temps écoulé depuis.	La mortalité dépend d'abord de l'âge réel de la personne, pas de l'âge qu'elle avait à la souscription. C'est la variable la plus importante d'une table.	Reconnu. Elle fait partie du socle
PolGrp	Le type de police, regroupé en six grandes familles : vie entière, temporaire, vie universelle, etc.	Les produits attirent des profils différents et n'ont pas les mêmes règles de souscription. Le regroupement évite des catégories trop petites pour être fiables.	Reconnu (la table distingue déjà les produits)
IssueYear	L'année où le contrat a été souscrit.	Les assurés d'une même génération ont été acceptés dans les mêmes conditions, avec les mêmes critères médicaux et les mêmes tests. Deux générations n'ont pas forcément le même niveau de sélection.	Notre idée
log_qx_amt	Le niveau de risque que la table attribue au groupe : le montant de décès attendu pour chaque dollar de capital exposé, exprimé en logarithme.	C'est la base de tout calcul de « réel sur prévu ». Elle indique au modèle à quel point la table juge le groupe risqué. L'échelle logarithmique rend les écarts comparables entre petits et grands taux.	Reconnu (la base « attendue » de toute étude)
log_avg_size	Le capital moyen d'un contrat dans le groupe, en logarithme.	Plus le capital est élevé, plus l'écart entre le réel et la table change nettement. Le capital sert d'indicateur du profil de l'assuré et de la rigueur de la souscription.	Reconnu (études d'expérience de la Société des actuaires)
cv2	Une mesure de l'inégalité des capitaux dans le groupe. Elle est proche de un quand tous les contrats sont semblables. Elle est plus élevée quand quelques gros contrats dominent.	Un seul décès sur un très gros contrat peut faire basculer le montant total. Ces groupes sont plus instables, et le modèle doit s'en méfier. Les actuaires traitent le même problème en plafonnant les gros contrats.	Concept reconnu. La variable est de notre conception
log_expos	Le nombre de polices sous risque dans le groupe, en logarithme.	Plus le volume est grand, plus les résultats sont stables. Cette variable permet au modèle d'accorder moins de poids aux groupes à faible volume, comme le fait la crédibilité.	Reconnu (l'exposition pondère la méthode de Bühlmann-Straub)
coh_ae_amt	La « mémoire du groupe » : l'écart entre le réel et le prévu observé chez les autres lignes du même groupe d'assurés (même sexe, statut de fumeur, type de police, capital, âge et année de souscription) sur les autres années. Elle est lissée quand les données sont rares. Elle est calculée sans jamais utiliser la ligne elle-même.	Un groupe qui a coûté plus, ou moins, que prévu a tendance à continuer. C'est le principe d'utiliser l'expérience propre d'un risque pour ajuster la prévision.	Notre idée (inspirée de Bühlmann)
coh_log_E	La quantité d'information qui se trouve derrière la mémoire du groupe, en logarithme.	Une mémoire construite sur beaucoup d'observations est fiable, une mémoire construite sur peu ne l'est pas. Cette variable permet au modèle de faire la différence.	Notr

Diapo 7 · Le tri (30 s, ex-8, simplifiée)

« Plus de variables, ce n'est pas mieux : certaines n'apportent rien et brouillent la prévision. Nous demandons donc au modèle lui-même : qu'est-ce qui t'a vraiment servi ? Ce qui a peu servi est écarté. Les fondamentaux, sexe, tabac, âge, durée et capital, restent toujours. »
Diapo 8 · L'évolutivité (50 s)
Reste l'avenir. Nous ne demandons pas au modèle de le deviner : il apprend la situation d'aujourd'hui. Pour les années à venir, nous ajoutons une hypothèse séparée, simple et visible : la mortalité continue de s'améliorer lentement, au rythme publié par l'Institut canadien des actuaires. Le chiffre est à l'écran.
