import open3d as o3d
import numpy as np
import os

DATA_DIR = 'res'
RST_FILE = 'zhu_rst.csv'
HELP = """
        程序使用说明
        Q: 下一个
        A: 正常
        Z: 噪音过多
        X: 明显孔洞
        C: 明显缺失
        S: 切换分割视图 （有分割文件才可以）
        D: 分割标注错误
        space: 切换背景 （绿色背景，更容易看出问题）
        .: 强制中断程序
        """


def main():
    print(HELP)
    check_files = ['clean_100k.ply', 'real_100k.ply', 'noiseless.ply', 'mesh.ply']
    rst_file = open(RST_FILE, 'r+', encoding="utf-8")
    lines = rst_file.readlines()
    if len(lines) == 0:
        # write head
        rst_file.write('name, clean_100k.ply, real_100k.ply, noiseless.ply, mesh.ply, comment\n')
        rst_file.flush()
    existed_names = {line.split(',')[0] for line in lines[1:]}
    print(f"already checked: (cnt:{len(existed_names)})")
    print(existed_names)

    processing_items_list = []
    for item in os.listdir(DATA_DIR):
        sub_dir = os.path.join(DATA_DIR, item)
        if os.path.isdir(sub_dir) and item not in existed_names:
            # if int(item.split('_')[0]) % 2 == 0:
            #     # 跳过偶数
            #     continue
            processing_items_list.append(item)

    for i, item in enumerate(processing_items_list):
        print(f"[{i}/{len(processing_items_list)}]")
        sub_dir = os.path.join(DATA_DIR, item)
        print(f'Dealing {sub_dir}')
        if len(os.listdir(sub_dir)) != 10:
            print(f'{item} data is not complete')
            rst_file.write(f'{item}, , , , ,data not complete\n')
            rst_file.flush()
            continue
        all_note_str = []
        for check_file in check_files:
            raw_point_cloud = o3d.io.read_point_cloud(os.path.join(sub_dir, check_file))
            seg_pc = []
            # load seg annotation
            if check_file == 'clean_100k.ply':
                annotation_file = os.path.join(sub_dir, 'annotation.txt')
                if not os.path.isfile(annotation_file):
                    print('annotation file not found')
                else:
                    with open(os.path.join(sub_dir, 'annotation.txt'), 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    temp_dic = {}
                    for line in lines:
                        k, v = line.replace('\n', '').split(' ')
                        if v not in temp_dic:
                            temp_dic[v] = []
                        temp_dic[v].append(int(k))
                    for k in temp_dic:
                        pc = raw_point_cloud.select_down_sample(temp_dic[k])
                        pc = pc.paint_uniform_color(np.random.rand(3, 1))
                        seg_pc.append(pc)
            rst_notes_set = set()
            VisualChecker(raw_point_cloud, seg_pc, rst_notes_set, window_name=f'{item} {check_file}').run()
            notes_str = ' '.join(rst_notes_set)
            all_note_str.append(notes_str)
            print(f"{item} {check_file} {notes_str}")
        if len(all_note_str) > 0:
            rst_file.write(f"{item},")
            rst_file.write(','.join(all_note_str))
            rst_file.write(',\n')
            rst_file.flush()


class VisualChecker:
    def __init__(self, pc, seg_pcs: list, rst_notes_set, window_name='open3d', width=960, height=960):
        self.pc = pc
        self.seg_pcs = seg_pcs
        self.rst_notes_set: set = rst_notes_set
        self.window_name = window_name
        self.width = width
        self.height = height
        self.bg_white_status = True
        self.show_seg_status = False

        self.key_to_call_back = {
            ord('A'): self.note_good,
            ord('Z'): self.note_noise,
            ord('X'): self.note_hole,
            ord('C'): self.note_imcomplete,
            ord('S'): self.toggle_seg,
            ord('D'): self.seg_wrong,
            ord(' '): self.change_background_color,
            ord('.'): self.abort
        }

    def run(self):
        o3d.visualization.draw_geometries_with_key_callbacks([self.pc], self.key_to_call_back, self.window_name,
                                                             width=self.width,
                                                             height=self.height)

    def change_background_color(self, vis: o3d.visualization.Visualizer):

        opt = vis.get_render_option()
        if self.bg_white_status:
            opt.background_color = np.asarray([0, 255, 0])
        else:
            opt.background_color = np.asarray([255, 255, 255])
        self.bg_white_status = not self.bg_white_status
        return False

    def abort(self, vis):
        print('abort')
        exit(-1)

    def toggle_seg(self, vis: o3d.visualization.Visualizer):
        if self.seg_pcs is None or len(self.seg_pcs) == 0:
            return
        if not self.show_seg_status:
            self.show_seg_status = True
            vis.clear_geometries()
            for pc in self.seg_pcs:
                vis.add_geometry(pc)
            vis.update_renderer()
        else:
            self.show_seg_status = False
            vis.clear_geometries()
            vis.add_geometry(self.pc)
            vis.update_renderer()

    def seg_wrong(self, vis):
        self.toggle_note('分割标注错误')

    def note_good(self, vis):
        self.toggle_note('正常')

    def note_noise(self, vis):
        self.toggle_note('噪音过多')

    def note_hole(self, vis):
        self.toggle_note('明显孔洞')

    def note_imcomplete(self, vis):
        self.toggle_note('明显缺失')

    def toggle_note(self, note):
        if note in self.rst_notes_set:
            self.rst_notes_set.remove(note)
        else:
            self.rst_notes_set.add(note)
        print(self.rst_notes_set)


if __name__ == '__main__':
    main()
