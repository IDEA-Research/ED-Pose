import os
import os.path as osp
import shutil
import time
import datetime

import torch

from util.config import Config

class Error(OSError):
    pass

def copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copyfile,
             ignore_dangling_symlinks=False):
    errors = []
    if os.path.isdir(src):
        names = os.listdir(src)
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()

        os.makedirs(dst)
        for name in names:
            if name in ignored_names:
                continue
            srcname = os.path.join(src, name)
            dstname = os.path.join(dst, name)
            try:
                if os.path.islink(srcname):
                    linkto = os.readlink(srcname)
                    if symlinks:
                        os.symlink(linkto, dstname)
                    else:
                        if not os.path.exists(linkto) and ignore_dangling_symlinks:
                            continue
                        if os.path.isdir(srcname):
                            copytree(srcname, dstname, symlinks, ignore,
                                    copy_function)
                        else:
                            copy_function(srcname, dstname)
                elif os.path.isdir(srcname):
                    copytree(srcname, dstname, symlinks, ignore, copy_function)
                else:
                    copy_function(srcname, dstname)
            except Error as err:
                errors.extend(err.args[0])
            except OSError as why:
                errors.append((srcname, dstname, str(why)))
    else:
        copy_function(src, dst)

    if errors:
        raise Error(errors)
    return dst

def check_and_copy(src_path, tgt_path):
    if os.path.exists(tgt_path):
        return None

    return copytree(src_path, tgt_path)


def remove(srcpath):
    if os.path.isdir(srcpath):
        return shutil.rmtree(srcpath)
    else:
        return os.remove(srcpath)  


def preparing_dataset(pathdict, image_set, args):
    start_time = time.time()
    dataset_file = args.dataset_file
    data_static_info = Config.fromfile('util/static_data_path.py')
    static_dict = data_static_info[dataset_file][image_set]

    copyfilelist = []
    for k,tgt_v in pathdict.items():
        if os.path.exists(tgt_v):
            if args.local_rank == 0:
                print("path <{}> exist. remove it!".format(tgt_v))
                remove(tgt_v)
            # continue
        
        if args.local_rank == 0:
            src_v = static_dict[k]
            assert isinstance(src_v, str)
            if src_v.endswith('.zip'):
                # copy
                cp_tgt_dir = os.path.dirname(tgt_v)
                filename = os.path.basename(src_v)
                cp_tgt_path = os.path.join(cp_tgt_dir, filename)
                print('Copy from <{}> to <{}>.'.format(src_v, cp_tgt_path))
                os.makedirs(cp_tgt_dir, exist_ok=True)
                check_and_copy(src_v, cp_tgt_path)          

                # unzip
                import zipfile
                print("Starting unzip <{}>".format(cp_tgt_path))
                with zipfile.ZipFile(cp_tgt_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(cp_tgt_path))      

                copyfilelist.append(cp_tgt_path)
                copyfilelist.append(tgt_v)
            else:
                print('Copy from <{}> to <{}>.'.format(src_v, tgt_v))
                os.makedirs(os.path.dirname(tgt_v), exist_ok=True)
                check_and_copy(src_v, tgt_v)
                copyfilelist.append(tgt_v)
    
    if len(copyfilelist) == 0:
        copyfilelist = None
    args.copyfilelist = copyfilelist
        
    if args.distributed:
        torch.distributed.barrier()
    total_time = time.time() - start_time
    if copyfilelist:
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Data copy time {}'.format(total_time_str))
    return copyfilelist


    