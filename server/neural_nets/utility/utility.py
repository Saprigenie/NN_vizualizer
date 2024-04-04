import torch
import os
import shutil


def get_children(model: torch.nn.Module):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def create_batch(dataset, start_i, batch_size):
    # Особый случай, батч вышел за пределы датасета.
    if start_i + batch_size > len(dataset):
        end_i = len(dataset)
        x = [0] * (len(dataset) - start_i)
        y = [0] * (len(dataset) - start_i)
    else:
        end_i = start_i + batch_size
        x = [0] * batch_size
        y = [0] * batch_size

    for i in range(start_i, end_i):
        x[i - start_i] = dataset[i][0].unsqueeze(0)
        y[i - start_i] = dataset[i][1].unsqueeze(0)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    return (x, y)


def remove_folder_content(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))