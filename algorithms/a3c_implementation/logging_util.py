import os
from tbparse import SummaryReader #!pip install tbparse
from torch.utils.tensorboard import SummaryWriter

def convert_all_logs_to_single_file(input_dir: str, output_dir: str):
    """
    Combines multiple events.out.tfevents files into a single file 

    A3C implementation currently results in multiple events.out.tfevents files (due to a work around addressing the multiprocessing pickling error)
    when training for 200k timesteps the number of events.out.tfevents files exceeds the buffer size of the tensorboard UI ;
    so this function merges the multiple files into a single file ready for display in the tensorboard UI

    Params:
        input_dir: the path to the folder which currently stores the multiple events.out.tfevents files
        output_dir: folder path for the resulting single events.out.tfevent file to go in
    """
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    reader = SummaryReader(input_dir)
    tensorboard_writer = SummaryWriter(output_dir)
    # extract scalars
    df_scalars = reader.scalars
    for index, row in df_scalars.iterrows():
        tensorboard_writer.add_scalar(row['tag'], row['value'], int(row['step']))

    # extract tensors
    df_tensors = reader.tensors
    for index, row in df_tensors.iterrows():
        tensorboard_writer.add_text(row['tag'], str(row['value']), int(row['step']))

    tensorboard_writer.close()
    print(f"Logs compiled into directory: {output_dir}")