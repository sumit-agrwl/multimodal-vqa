"""
This file provide data loading for phase 1 development.
We could ground truth scene graph from this file.

Need to refactor the graph loading components out.
"""
import json
import torch
import numpy as np
import torch_geometric

from fine.constants import constants
from fine.graphvqa.config import graphVQAConfig
from utils import load_obj_tsv, load_obj_tsv_by_id
from fine.vocab.vocab_utils import SGVocab, QAVocab


ROOT_DIR = constants.ROOT_DIR
CONFIG_FILE = constants.CONFIG_FILE
SPLIT_TO_PROGRAMMED_QUESTION_PATH_TABLE = constants.SPLIT_TO_PROGRAMMED_QUESTION_PATH_TABLE
SPLIT_TO_SCENE_GRAPH_PATH_TABLE = constants.SPLIT_TO_SCENE_GRAPH_PATH_TABLE

class SGFeature():
    # GT Scene Graph Vocab: to preprocess and numerlicalize text
    def translate(self, sg, new_execution_buffer: list):
        ##################################
        # handle scene graph part
        ##################################
        sg_datum = self.convert_one_gqa_scene_graph(sg)

        ##################################
        # Do translation for target object IDs in th execution buffer
        # No need to translate. already based on sorted sg obj_id.
        # TODO: empty object mapping - fix that.
        ##################################

        execution_bitmap = torch.zeros(
            (sg_datum.num_nodes, graphVQAConfig.max_execution_step),
            dtype=torch.float32
        )
        instr_annotated_len = min(
            len(new_execution_buffer), graphVQAConfig.max_execution_step
        )
        padding_len = graphVQAConfig.max_execution_step - instr_annotated_len

        ##################################
        # Build Bitmap based on instructions
        ##################################
        for instr_idx in range(instr_annotated_len):
            execution_target_list = new_execution_buffer[instr_idx]
            for trans_obj_id in execution_target_list:
                execution_bitmap[trans_obj_id, instr_idx] = 1.0

        ##################################
        # Padding Bitmap by copying the last annotation
        ##################################
        for instr_idx in range(instr_annotated_len, instr_annotated_len + padding_len):
            execution_bitmap[:, instr_idx] = execution_bitmap[:, instr_annotated_len - 1]

        sg_datum.y = execution_bitmap

        return sg_datum

    def convert_one_gqa_scene_graph(self, sg_this):
        ##################################
        # Make sure that it is not an empty graph
        ##################################
        # assert len(sg_this['objects'].keys()) != 0, sg_this
        if len(sg_this['objects'].keys()) == 0:
            # only in val
            # print("Got Empty Scene Graph", sg_this) # only one empty scene graph during val
            # use a dummy scene graph instead
            sg_this = {
                'objects': {
                    '0': {
                        'name': '<UNK>',
                        'relations': [
                            {
                                'object': '1',
                                'name': '<UNK>',
                            }
                        ],
                        'attributes': ['<UNK>'],
                    },
                    '1': {
                        'name': '<UNK>',
                        'relations': [
                            {
                                'object': '0',
                                'name': '<UNK>',
                            }
                        ],
                        'attributes': ['<UNK>'],
                    },

                }
            }

        ##################################
        # graph node: objects
        ##################################
        objIDs = sorted(sg_this['objects'].keys()) # str
        map_objID_to_node_idx = {objID: node_idx for node_idx, objID in enumerate(objIDs)}

        ##################################
        # Initialize Three key components for graph representation
        ##################################
        node_feature_list = []
        edge_feature_list = []
        # [[from, to], ...]
        edge_topology_list = []
        added_sym_edge_list = [] # yanhao: record the index of added edges in the edge_feature_list

        ##################################
        # Duplicate edges, making sure that the topology is symmetric
        ##################################
        from_to_connections_set = set()
        for node_idx in range(len(objIDs)):
            objId = objIDs[node_idx]
            obj = sg_this['objects'][objId]
            for rel in obj['relations']:
                # [from self as source, to outgoing]
                from_to_connections_set.add((node_idx, map_objID_to_node_idx[rel["object"]]))
        # print("from_to_connections_set", from_to_connections_set)

        for node_idx in range(len(objIDs)):
            ##################################
            # Traverse Scene Graph's objects based on node idx order
            ##################################
            objId = objIDs[node_idx]
            obj = sg_this['objects'][objId]

            ##################################
            # Encode Node Feature: object category, attributes
            # Note: not encoding spatial information
            # - obj['x'], obj['y'], obj['w'], obj['h']
            ##################################
            # MAX_OBJ_TOKEN_LEN = 4 # 1 name + 3 attributes
            MAX_OBJ_TOKEN_LEN = 12

            # 4 X '<pad>'
            object_token_arr = np.ones(MAX_OBJ_TOKEN_LEN, dtype=np.int_) * \
                SGVocab.sg_encoding_text.vocab.stoi[SGVocab.sg_encoding_text.pad_token]

            # should have no error
            object_token_arr[0] = SGVocab.sg_encoding_text.vocab.stoi[obj['name']]
            # assert object_token_arr[0] !=0 , obj
            if object_token_arr[0] == 0:
                # print("Out Of Vocabulary Object:", obj['name'])
                pass
            # now use this constraint: 1–3 attributes
            # deduplicate

            ##################################
            # Comment out this to see the importance of attributes
            ##################################

            for attr_idx, attr in enumerate(set(obj['attributes'])):
                object_token_arr[attr_idx + 1] = SGVocab.sg_encoding_text.vocab.stoi[attr]

            node_feature_list.append(object_token_arr)

            ##################################
            # Need to Add a self-looping edge
            ##################################
            edge_topology_list.append([node_idx, node_idx]) # [from self, to self]
            edge_token_arr = np.array([SGVocab.sg_encoding_text.vocab.stoi['<self>']], dtype=np.int_)
            edge_feature_list.append(edge_token_arr)

            ##################################
            # Encode Edge
            # - Edge Feature: edge label (name)
            # - Edge Topology: adjacency matrix
            # GQA relations [dict]  A list of all outgoing relations (edges) from the object (source).
            ##################################


            ##################################
            # Comment out the whole for loop to see the importance of attributes
            ##################################

            for rel in obj['relations']:
                # [from self as source, to outgoing]
                edge_topology_list.append([node_idx, map_objID_to_node_idx[rel["object"]]])
                # name of the relationship
                edge_token_arr = np.array([SGVocab.sg_encoding_text.vocab.stoi[rel["name"]]], dtype=np.int_)
                edge_feature_list.append(edge_token_arr)

                ##################################
                # Symmetric
                # - If there is no symmetric edge, add one.
                # - Should add mechanism to check duplicates
                ##################################
                if (map_objID_to_node_idx[rel["object"]], node_idx) not in from_to_connections_set:
                    # print("catch!", (map_objID_to_node_idx[rel["object"]], node_idx), rel["name"])

                    # reverse of [from self as source, to outgoing]
                    edge_topology_list.append([map_objID_to_node_idx[rel["object"]], node_idx])
                    # re-using name of the relationship
                    edge_feature_list.append(edge_token_arr) 

                    # yanhao: record the added edge's index in feature and idx array:
                    added_sym_edge_list.append(len(edge_feature_list)-1)

        ##################################
        # Convert to standard pytorch geometric format
        # - node_feature_list
        # - edge_feature_list
        # - edge_topology_list
        ##################################

        # print("sg_this", sg_this)
        # print("objIDs", objIDs)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        # print("node_feature_list_arr", node_feature_list_arr.shape)

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        # print("edge_feature_list_arr", edge_feature_list_arr.shape)

        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        # print("edge_topology_list_arr", edge_topology_list_arr.shape)
        del edge_topology_list_arr

        # edge_index = torch.tensor([[0, 1],
        #                         [1, 0],
        #                         [1, 2],
        #                         [2, 1]], dtype=torch.long)
        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()
        datum = torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

        # yanhao: add an additional variable to datum:
        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge

        return datum


class GQADataset:
    def __init__(self, split: str):
        self.split = split

        # Loading datasets to data
        self.data = []
        data_file_name = SPLIT_TO_PROGRAMMED_QUESTION_PATH_TABLE[split]
        self.data.extend(json.load(open(ROOT_DIR / data_file_name)))
        print("Load %d data from split(s) %s." % (len(self.data), self.split))
        #print (self.data)
        # List to dict (for evaluation and others)
        self.id2datum = {
            datum[3]: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(ROOT_DIR / "meta_info/trainval_ans2label.json"))
        self.label2ans = json.load(open(ROOT_DIR / "meta_info/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():
    def __init__(self):
        self.key2data = {}
        self.id2Data = {}
        self.idData = []
    def load_data(self, name, number):
        if name == 'testdev':
            path = ROOT_DIR / "imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = ROOT_DIR / "imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]

    def load_id_data(self, name, ids):
        if name == 'testdev':
            path = ROOT_DIR / "imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = ROOT_DIR / "imgfeat/vg_gqa_obj36.tsv"
        pickUpIds = []
        self.id2Data = []
        for id in ids:
            if id not in self.id2Data:
                pickUpIds.append(id)
            else:
                self.idData.append(self.id2Data[id])
        
        self.idData.extend(load_obj_tsv_by_id(path, pickUpIds))
        return self.idData


gqa_buffer_loader = GQABufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset
        

        self.imgid2sg = []
        if 'testdev' in dataset.split or 'testdev_all' in dataset.split: 
            self.imgid2sg = []
        else:
            self.imgid2sg = json.load(open(SPLIT_TO_SCENE_GRAPH_PATH_TABLE[dataset.split]))
        
        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        #img_ids = list(json.load(open(SPLIT_TO_IMG_PATH_TABLE[dataset.split])))
        if 'testdev' in dataset.split or 'testdev_all' in dataset.split:     # Always loading all the data in testdev
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', -1))
        
        self.sg_feature_extractor = SGFeature()

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        
        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum[0] in self.imgid2img: #datum[0] is image_id
                self.data.append(datum)
        #print (self.data)
        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # Get question info
        datum = self.data[index]
        img_id = datum[0]
        ques_id = datum[3]
        ques = datum[1]
        # Get fine question feature
        ques_fine = QAVocab.qa_encoding_text.preprocess(datum[1])
        
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Prepare label
        answer = datum[4]
        new_execution_buffer = datum[8]
        new_programs_hierarchical_decoder = datum[9]
        types = datum[10]

        if answer == 'bottle cap':
            answer = 'bottle'
        assert answer in self.raw_dataset.ans2label, answer
        label = self.raw_dataset.ans2label[answer]

        # get scene graph info
        sg = None
        if self.raw_dataset.split != 'test':
            sg = self.sg_feature_extractor.translate(self.imgid2sg[img_id], new_execution_buffer)
        
        # get programs and pad them to max_execution_step
        prog = new_programs_hierarchical_decoder[: graphVQAConfig.max_execution_step]
        prog += (graphVQAConfig.max_execution_step - len(prog)) * [[]]
        return (
            ques_id, feats, boxes, ques, ques_fine, sg, prog, label, types
        )

    @property
    def num_answers(self):
        return len(self.raw_dataset.ans2label)

    @classmethod
    def indices_to_string(cls, indices, words=False):
        """Convert word indices (torch.Tensor) to sentence (string).
        Args:
            indices: torch.tensor or numpy.array of shape (T) or (T, 1)
            words: boolean, wheter return list of words
        Returns:
            sentence: string type of converted sentence
            words: (optional) list[string] type of words list
        """
        sentence = list()
        for idx in indices:
            word = QAVocab.qa_encoding_text.vocab.itos[idx.item()]

            if word in ["<pad>", "<start>"]:
                continue
            if word in ["<end>"]:
                break

            # no needs of space between the special symbols
            if len(sentence) and word in ["'", ".", "?", "!", ","]:
                sentence[-1] += word
            else:
                sentence.append(word)

        if words:
            return " ".join(sentence), sentence
        return " ".join(sentence)


def GQATorchDataset_collate_fn(data):
    ques_id, feats, boxes, ques, ques_fine, sg, prog, label, types = zip(*data)
    ques_fine_processed = QAVocab.qa_encoding_text.process(ques_fine)
    #sg_processed = torch_geometric.data.Batch.from_data_list(sg)
    sg_processed = sg
    expand_prog_processed = []
    for itrs in prog:
        expand_prog_processed.extend(itrs)
    assert len(expand_prog_processed) % graphVQAConfig.max_execution_step == 0, expand_prog_processed
    prog_processed = QAVocab.qa_encoding_text.process(expand_prog_processed)

    '''
    #prog_processed.reshape((prog_processed.size(0), graphVQAConfig.max_execution_step, prog_processed.size(1)//graphVQAConfig.max_execution_step))
    # no_tokens, max_step*batch_size
    max_step = graphVQAConfig.max_execution_step
    bs = prog_processed.size(1) // max_step
    n_tokens = prog_processed.size(0)
    # max_step*batch_size, no_tokens
    prog_processed = prog_processed.permute(1, 0)
    
    prog_processed = prog_processed.reshape((bs, max_step, n_tokens))
    prog_processed = prog_processed.reshape((bs, max_step*n_tokens))
    # no_tokens * max_step, batch
    prog_processed = prog_processed.permute(1, 0)
    #print (prog_processed.size())
    '''

    return (
        ques_id, torch.Tensor(feats), torch.Tensor(boxes), ques, torch.LongTensor(ques_fine_processed),
        sg_processed, torch.LongTensor(prog_processed), torch.LongTensor(label), types
    )
