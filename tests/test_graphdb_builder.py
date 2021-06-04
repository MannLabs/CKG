import urllib
from ckg.graphdb_builder import builder_utils

class TestParsersClass:
    def get_url_response(self, url):
        url = url.replace('ftp:', 'http:')
        response = urllib.request.urlopen(url).getcode()
        
        return response
            
    
    def get_config(self, database):
        config = builder_utils.get_config(config_name="{}Config.yml".format(database), data_type='databases')
        
        return config
    
    def test_uniprot_url(self):
        config = self.get_config("uniprot")
        url_ids = ["uniprot_id_url", 
                    "uniprot_variant_file", 
                    "uniprot_go_annotations", 
                    "uniprot_fasta_file", 
                    "release_notes", 
                    "uniprot_peptides_files"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                if isinstance(url, list):
                    for u in url:
                        print("Testing url", u, self.get_url_response(u))
                        assert self.get_url_response(u) == 200
                else: 
                    print("Testing url", url, self.get_url_response(url))
                    assert self.get_url_response(url) == 200
                    
    def test_textmining_url(self):
        config = self.get_config("jensenlab")
        url_id = config['db_url']
        files = ["organisms_file",
                 "db_mentions_files"]
        
        for fid in files:
            filename = config[fid]
            if isinstance(filename, dict):
                for ftype in filename:
                    fname = filename[ftype]
                    url = url_id.replace("FILE", fname)
                    print("Testing url", url, self.get_url_response(url))
                    assert self.get_url_response(url) == 200
            else:
                url = config['db_url'].replace("FILE", filename)
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200

    def test_STRING_url(self):
        config = self.get_config("string")
        url_ids =  ["STRING_mapping_url",
                    "STRING_url", 
                    "STRING_actions_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_STITCH_url(self):
        config = self.get_config("string")
        url_ids =  ["STITCH_mapping_url",
                    "STITCH_url", 
                    "STITCH_actions_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_GWASCatalog_url(self):
        config = self.get_config("gwasCatalog")
        url_ids =  ["GWASCat_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_CORUM_url(self):
        config = self.get_config("corum")
        url_ids =  ["database_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_disgenet_url(self):
        config = self.get_config("disgenet")
        url_id = config['disgenet_url']
        files = config['disgenet_files']
        files.update(config['disgenet_mapping_files'])

        for ftype in files:
            fname = files[ftype]
            url = url_id + fname
            print("Testing url", url, self.get_url_response(url))
            assert self.get_url_response(url) == 200

    
    def test_DGIDB_url(self):
        config = self.get_config("drugGeneInteractionDB")
        url_ids =  ["DGIdb_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
    def test_HGNC_url(self):
        config = self.get_config("hgnc")
        url_ids =  ["hgnc_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_HMDB_url(self):
        config = self.get_config("hmdb")
        url_ids =  ["HMDB_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_HPA_url(self):
        config = self.get_config("hpa")
        url_ids =  ["hpa_pathology_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_Intact_url(self):
        config = self.get_config("intact")
        url_ids =  ["intact_psimitab_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
    
    def test_MutationDS_url(self):
        config = self.get_config("mutationDS")
        url_ids =  ["mutations_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
    
    def test_OncoKB_url(self):
        config = self.get_config("oncokb")
        url_ids =  ["OncoKB_annotated_url", 
                    "OncoKB_actionable_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_Pfam_url(self):
        config = self.get_config("pfam")
        url_id = config['ftp_url']
        files = ["full_uniprot_file"]
        
        for fid in files:
            filename = config[fid]
            url = url_id + filename
            print("Testing url", url, self.get_url_response(url))
            assert self.get_url_response(url) == 200
            
            
    def test_Reactome_url(self):
        config = self.get_config("reactome")
        url_ids =  config["reactome_urls"]
        
        for url_id in url_ids:
            url = url_ids[url_id]
            print("Testing url", url, self.get_url_response(url))
            assert self.get_url_response(url) == 200
            
            
    def test_Refseq_url(self):
        config = self.get_config("refseq")
        url_ids =  ["refseq_url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_Sider_url(self):
        config = self.get_config("sider")
        url_ids =  ["SIDER_url",
                    "SIDER_indications"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_Signor_url(self):
        config = self.get_config("signor")
        url_ids =  ["url"]
        
        for url_id in url_ids:
            if url_id in config:
                url = config[url_id]
                print("Testing url", url, self.get_url_response(url))
                assert self.get_url_response(url) == 200
                
                
    def test_SMPDB_url(self):
        config = self.get_config("smpdb")
        url_ids =  config["smpdb_urls"]
        
        for url_id in url_ids:
            url = url_ids[url_id]
            print("Testing url", url, self.get_url_response(url))
            assert self.get_url_response(url) == 200