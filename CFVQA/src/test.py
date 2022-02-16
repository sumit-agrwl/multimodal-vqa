import torch
from gqa.gqa_data import GQATorchDataset
from fine.graphvqa.modeling import PipelineModel
from fine.vocab.vocab_utils import QAVocab, SGVocab

if __name__ == "__main__":

    ##################################
    # Need to have the vocab first to debug
    ##################################
    from fine.vocab.vocab_utils import TextProcessor, Vocab
    qaVocab = QAVocab(build=False)
    sgVocab = SGVocab(build=False)
    debug_dataset = GQATorchDataset(
        split='val_unbiased'
    )
    
    model = PipelineModel(qaVocab.qa_encoding_text)
    model.train()

    ##################################
    # Simulate Batching
    ##################################

    data_loader = torch.utils.data.DataLoader(debug_dataset, batch_size=2, shuffle=True, num_workers=0)
    for data_batch in data_loader:
        # print("data_batch", data_batch)
        questionID, questions, gt_scene_graphs, programs, full_answers, short_answer_label, types = data_batch
        print("gt_scene_graphs", gt_scene_graphs)
        # print("gt_scene_graphs.x", gt_scene_graphs.x)
        # print("gt_scene_graphs.edge_index[0]", gt_scene_graphs.edge_index[0])
        # print("gt_scene_graphs.edge_attr", gt_scene_graphs.edge_attr )
        # print(gt_scene_graphs.batch)


        ##################################
        # Prepare training input and training target for text generation
        # - shape [len, batch]
        ##################################
        programs_input = programs[:-1]
        programs_target = programs[1:]
        full_answers_input = full_answers[:-1]
        full_answers_target = full_answers[1:]

        output = model(
            questions,
            gt_scene_graphs,
            programs_input,
            full_answers_input
        )

        print("model output:", output)
        break