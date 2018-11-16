#### SIDER database #####
SIDER_url = "http://sideeffects.embl.de/media/download/meddra_all_label_se.tsv.gz"
SIDER_mapping = "http://stitch.embl.de/download/chemical.aliases.v5.0.tsv.gz"
SIDER_indications = "http://sideeffects.embl.de/media/download/meddra_all_indications.tsv.gz"
SIDER_source = "UMLS"
outputfileName = "sider_has_side_effect.csv"
indications_outputfileName = "sider_is_indicated_for.csv"
header = ['START_ID', 'END_ID','TYPE', 'source', 'original_side_effect']
indications_header = ['START_ID', 'END_ID','TYPE', 'evidence', 'source', 'original_side_effect']
