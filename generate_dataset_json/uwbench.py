import json
import os
from pathlib import Path


DEFAULT_UWBENCH_ROOT = Path(__file__).resolve().parents[2] / 'UW-Bench' / 'training_set'


class UWBenchSolver(object):
    def __init__(self, root=DEFAULT_UWBENCH_ROOT, meta_path=None):
        self.root = Path(root)
        default_meta_path = Path(__file__).resolve().parents[1] / 'data_meta' / 'uwbench_meta.json'
        self.meta_path = Path(meta_path) if meta_path is not None else default_meta_path

    def _load_split(self, split_name):
        label_path = self.root / 'labels' / f'uw_bench_detection_{split_name}.json'
        with open(label_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        cls_info = []
        anomaly_samples = 0
        normal_samples = 0
        for record in records:
            img_path = record['images'][0].replace('\\', '/')
            stem = Path(img_path).stem
            mask_path = f'SegmentationClass/{stem}.png'
            anomaly = 0 if record['output'].strip() == '[]' else 1
            cls_info.append(dict(
                img_path=img_path,
                mask_path=mask_path if anomaly else '',
                cls_name='road',
                specie_name='water' if anomaly else 'waterless',
                anomaly=anomaly,
            ))
            if anomaly:
                anomaly_samples += 1
            else:
                normal_samples += 1
        return cls_info, normal_samples, anomaly_samples

    def run(self):
        info = dict(train={}, test={})
        train_info, train_normal, train_anomaly = self._load_split('train')
        test_info, test_normal, test_anomaly = self._load_split('val')
        info['train']['road'] = train_info
        info['test']['road'] = test_info

        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(info, indent=4) + '\n')

        print(f'wrote {self.meta_path}')
        print('train normal_samples', train_normal, 'anomaly_samples', train_anomaly)
        print('test normal_samples', test_normal, 'anomaly_samples', test_anomaly)


if __name__ == '__main__':
    root = os.environ.get('UWBENCH_ROOT', DEFAULT_UWBENCH_ROOT)
    meta_path = os.environ.get('UWBENCH_META_PATH')
    UWBenchSolver(root=root, meta_path=meta_path).run()
