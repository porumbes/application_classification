#!/bin/bash

DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/application_classification/"
DATA_SUB[0]="ac_JohnsHopkins_random"

NAME[0]="rmat18"
NAME[1]="georgiyPattern"
NAME[2]="JohnsHopkins"

python prob2bin.py \
	${DATA_PREFIX}${NAME[0]}/${NAME[0]}.Vertex.csv ${DATA_PREFIX}${NAME[0]}/${NAME[0]}.Edges.csv \
      	./data/${NAME[1]}.Vertex.csv ./data/${NAME[1]}.Edges.csv \
	--out_data_prefix ${NAME[0]} --out_pattern_prefix ${NAME[1]}

python prob2bin.py \
	${DATA_PREFIX}${DATA_SUB[0]}/${NAME[2]}.data.nodes ${DATA_PREFIX}${DATA_SUB[0]}/${NAME[2]}.data.edges \
	${DATA_PREFIX}${DATA_SUB[0]}/${NAME[2]}.pattern.nodes ${DATA_PREFIX}${DATA_SUB[0]}/${NAME[2]}.pattern.edges \
	--out_data_prefix ${NAME[2]} --out_pattern_prefix ${NAME[2]}
