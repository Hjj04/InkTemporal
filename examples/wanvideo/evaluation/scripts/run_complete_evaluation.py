#!/usr/bin/env python3
"""
完整的多级别视频质量评估脚本 - 修复版
适配现有评估脚本的参数接口
作者：Hanzhe
日期：2025-10-17
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import traceback


class VideoEvaluationPipeline:
    """视频评估流水线类"""
    
    def __init__(self, args):
        self.generated_videos_dir = args.videos_dir
        self.real_videos_dir = args.real_videos_dir
        self.output_dir = args.output_dir
        self.device = args.device
        self.levels = args.levels
        self.skip_existing = args.skip_existing
        self.temporal_checkpoint = args.temporal_checkpoint
        
        # 创建输出目录
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.summary_dir = os.path.join(self.output_dir, 'summary')
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        
        # 脚本目录
        self.scripts_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 评估结果存储
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'generated_videos_dir': self.generated_videos_dir,
                'real_videos_dir': self.real_videos_dir,
                'device': self.device,
                'levels': self.levels
            },
            'results_by_level': {},
            'summary': {}
        }
    
    def log(self, message, level='INFO'):
        """日志输出"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
        sys.stdout.flush()
    
    def check_file_exists(self, filepath):
        """检查文件是否存在"""
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0
    
    def get_level_video_dir(self, level):
        """获取特定级别的视频目录"""
        # 提取级别数字 (level_0 -> 0)
        level_num = level.split('_')[1] if '_' in level else level
        
        # 检查是否有子目录结构
        level_subdir = os.path.join(self.generated_videos_dir, level)
        if os.path.exists(level_subdir):
            return level_subdir
        
        # 如果没有子目录，返回主目录（稍后通过文件名过滤）
        return self.generated_videos_dir
    
    def evaluate_tlpips(self, level):
        """评估tLPIPS指标"""
        output_file = os.path.join(self.metrics_dir, f'tlpips_{level}.json')
        
        # 检查是否跳过
        if self.skip_existing and self.check_file_exists(output_file):
            self.log(f"跳过 tLPIPS ({level}) - 结果已存在", 'SKIP')
            return True, output_file
        
        self.log(f"开始评估 tLPIPS (Level: {level})")
        
        # tLPIPS脚本使用 --video_dir 参数
        video_dir = self.get_level_video_dir(level)
        script_path = os.path.join(self.scripts_dir, 'eval_tlpips.py')
        
        if not os.path.exists(script_path):
            self.log(f"脚本不存在: {script_path}", 'ERROR')
            return False, None
        
        cmd = [
            'python3',
            script_path,
            '--video_dir', video_dir,
            '--output_file', output_file,
            '--device', self.device
        ]
        
        self.log(f"执行命令: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed_time = time.time() - start_time
            
            self.log(f"完成 tLPIPS ({level}) - 耗时: {elapsed_time:.2f}秒")
            if result.stdout:
                self.log(f"输出:\n{result.stdout}")
            
            return True, output_file
            
        except subprocess.CalledProcessError as e:
            self.log(f"评估失败 tLPIPS ({level})", 'ERROR')
            self.log(f"错误: {e.stderr}", 'ERROR')
            return False, None
        except Exception as e:
            self.log(f"未知错误: {str(e)}", 'ERROR')
            traceback.print_exc()
            return False, None
    
    def evaluate_warping_error(self, level):
        """评估Warping Error指标"""
        output_file = os.path.join(self.metrics_dir, f'warping_error_{level}.json')
        
        # 检查是否跳过
        if self.skip_existing and self.check_file_exists(output_file):
            self.log(f"跳过 Warping Error ({level}) - 结果已存在", 'SKIP')
            return True, output_file
        
        self.log(f"开始评估 Warping Error (Level: {level})")
        
        # 检查temporal_checkpoint
        if not self.temporal_checkpoint or not os.path.exists(self.temporal_checkpoint):
            self.log(f"Temporal checkpoint 不存在: {self.temporal_checkpoint}", 'ERROR')
            return False, None
        
        video_dir = self.get_level_video_dir(level)
        script_path = os.path.join(self.scripts_dir, 'eval_warping_error.py')
        
        if not os.path.exists(script_path):
            self.log(f"脚本不存在: {script_path}", 'ERROR')
            return False, None
        
        cmd = [
            'python3',
            script_path,
            '--video_dir', video_dir,
            '--temporal_checkpoint', self.temporal_checkpoint,
            '--output_file', output_file,
            '--device', self.device
        ]
        
        self.log(f"执行命令: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed_time = time.time() - start_time
            
            self.log(f"完成 Warping Error ({level}) - 耗时: {elapsed_time:.2f}秒")
            if result.stdout:
                self.log(f"输出:\n{result.stdout}")
            
            return True, output_file
            
        except subprocess.CalledProcessError as e:
            self.log(f"评估失败 Warping Error ({level})", 'ERROR')
            self.log(f"错误: {e.stderr}", 'ERROR')
            return False, None
        except Exception as e:
            self.log(f"未知错误: {str(e)}", 'ERROR')
            traceback.print_exc()
            return False, None
    
    def evaluate_fid(self, level):
        """评估FID指标"""
        output_file = os.path.join(self.metrics_dir, f'fid_{level}.json')
        
        # 检查是否跳过
        if self.skip_existing and self.check_file_exists(output_file):
            self.log(f"跳过 FID ({level}) - 结果已存在", 'SKIP')
            return True, output_file
        
        self.log(f"开始评估 FID (Level: {level})")
        
        script_path = os.path.join(self.scripts_dir, 'eval_fid.py')
        
        if not os.path.exists(script_path):
            self.log(f"脚本不存在: {script_path}", 'ERROR')
            return False, None
        
        # FID使用 --real_data_dir
        cmd = [
            'python3',
            script_path,
            '--videos_dir', self.generated_videos_dir,
            '--real_data_dir', self.real_videos_dir,
            '--level', level,
            '--output_file', output_file,
            '--device', self.device
        ]
        
        self.log(f"执行命令: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed_time = time.time() - start_time
            
            self.log(f"完成 FID ({level}) - 耗时: {elapsed_time:.2f}秒")
            if result.stdout:
                self.log(f"输出:\n{result.stdout}")
            
            return True, output_file
            
        except subprocess.CalledProcessError as e:
            self.log(f"评估失败 FID ({level})", 'ERROR')
            self.log(f"错误: {e.stderr}", 'ERROR')
            return False, None
        except Exception as e:
            self.log(f"未知错误: {str(e)}", 'ERROR')
            traceback.print_exc()
            return False, None
    
    def evaluate_fvd(self, level):
        """评估FVD指标"""
        output_file = os.path.join(self.metrics_dir, f'fvd_{level}.json')
        
        # 检查是否跳过
        if self.skip_existing and self.check_file_exists(output_file):
            self.log(f"跳过 FVD ({level}) - 结果已存在", 'SKIP')
            return True, output_file
        
        self.log(f"开始评估 FVD (Level: {level})")
        
        script_path = os.path.join(self.scripts_dir, 'eval_fvd.py')
        
        if not os.path.exists(script_path):
            self.log(f"脚本不存在: {script_path}", 'ERROR')
            return False, None
        
        cmd = [
            'python3',
            script_path,
            '--videos_dir', self.generated_videos_dir,
            '--real_videos_dir', self.real_videos_dir,
            '--level', level,
            '--output_file', output_file,
            '--device', self.device
        ]
        
        self.log(f"执行命令: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed_time = time.time() - start_time
            
            self.log(f"完成 FVD ({level}) - 耗时: {elapsed_time:.2f}秒")
            if result.stdout:
                self.log(f"输出:\n{result.stdout}")
            
            return True, output_file
            
        except subprocess.CalledProcessError as e:
            self.log(f"评估失败 FVD ({level})", 'ERROR')
            self.log(f"错误: {e.stderr}", 'ERROR')
            return False, None
        except Exception as e:
            self.log(f"未知错误: {str(e)}", 'ERROR')
            traceback.print_exc()
            return False, None
    
    def evaluate_edge_iou(self, level):
        """评估Edge IoU指标"""
        output_file = os.path.join(self.metrics_dir, f'edge_iou_{level}.json')
        
        # 检查是否跳过
        if self.skip_existing and self.check_file_exists(output_file):
            self.log(f"跳过 Edge IoU ({level}) - 结果已存在", 'SKIP')
            return True, output_file
        
        self.log(f"开始评估 Edge IoU (Level: {level})")
        
        script_path = os.path.join(self.scripts_dir, 'eval_edge_iou.py')
        
        if not os.path.exists(script_path):
            self.log(f"脚本不存在: {script_path}", 'ERROR')
            return False, None
        
        # Edge IoU不需要 --real_videos_dir
        cmd = [
            'python3',
            script_path,
            '--videos_dir', self.generated_videos_dir,
            '--level', level,
            '--output_file', output_file,
            '--device', self.device
        ]
        
        self.log(f"执行命令: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed_time = time.time() - start_time
            
            self.log(f"完成 Edge IoU ({level}) - 耗时: {elapsed_time:.2f}秒")
            if result.stdout:
                self.log(f"输出:\n{result.stdout}")
            
            return True, output_file
            
        except subprocess.CalledProcessError as e:
            self.log(f"评估失败 Edge IoU ({level})", 'ERROR')
            self.log(f"错误: {e.stderr}", 'ERROR')
            return False, None
        except Exception as e:
            self.log(f"未知错误: {str(e)}", 'ERROR')
            traceback.print_exc()
            return False, None
    
    def load_metric_result(self, result_file):
        """加载评估结果"""
        if not result_file or not os.path.exists(result_file):
            return None
        
        try:
            with open(result_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"加载结果失败 {result_file}: {str(e)}", 'ERROR')
            return None
    
    def evaluate_single_level(self, level):
        """评估单个级别的所有指标"""
        self.log(f"\n{'='*80}")
        self.log(f"开始评估 {level}")
        self.log(f"{'='*80}\n")
        
        level_results = {
            'level': level,
            'metrics': {},
            'success': {},
            'errors': []
        }
        
        # 1. 评估tLPIPS
        success, result_file = self.evaluate_tlpips(level)
        level_results['success']['tlpips'] = success
        if success:
            metric_data = self.load_metric_result(result_file)
            if metric_data:
                level_results['metrics']['tlpips'] = metric_data
        else:
            level_results['errors'].append('tLPIPS evaluation failed')
        
        # 2. 评估Warping Error
        success, result_file = self.evaluate_warping_error(level)
        level_results['success']['warping_error'] = success
        if success:
            metric_data = self.load_metric_result(result_file)
            if metric_data:
                level_results['metrics']['warping_error'] = metric_data
        else:
            level_results['errors'].append('Warping Error evaluation failed')
        
        # 3. 评估FID
        success, result_file = self.evaluate_fid(level)
        level_results['success']['fid'] = success
        if success:
            metric_data = self.load_metric_result(result_file)
            if metric_data:
                level_results['metrics']['fid'] = metric_data
        else:
            level_results['errors'].append('FID evaluation failed')
        
        # 4. 评估FVD
        success, result_file = self.evaluate_fvd(level)
        level_results['success']['fvd'] = success
        if success:
            metric_data = self.load_metric_result(result_file)
            if metric_data:
                level_results['metrics']['fvd'] = metric_data
        else:
            level_results['errors'].append('FVD evaluation failed')
        
        # 5. 评估Edge IoU
        success, result_file = self.evaluate_edge_iou(level)
        level_results['success']['edge_iou'] = success
        if success:
            metric_data = self.load_metric_result(result_file)
            if metric_data:
                level_results['metrics']['edge_iou'] = metric_data
        else:
            level_results['errors'].append('Edge IoU evaluation failed')
        
        # 保存当前级别的结果
        level_summary_file = os.path.join(self.summary_dir, f'{level}_summary.json')
        with open(level_summary_file, 'w') as f:
            json.dump(level_results, f, indent=2)
        
        self.log(f"级别 {level} 评估完成，结果保存到: {level_summary_file}")
        
        return level_results
    
    def evaluate_all_levels(self):
        """评估所有级别"""
        self.log(f"\n{'#'*80}")
        self.log(f"开始完整评估流程")
        self.log(f"评估级别: {', '.join(self.levels)}")
        self.log(f"{'#'*80}\n")
        
        overall_start_time = time.time()
        
        for level in self.levels:
            try:
                level_results = self.evaluate_single_level(level)
                self.results['results_by_level'][level] = level_results
            except Exception as e:
                self.log(f"评估级别 {level} 时发生错误: {str(e)}", 'ERROR')
                traceback.print_exc()
                self.results['results_by_level'][level] = {
                    'level': level,
                    'error': str(e),
                    'success': False
                }
        
        overall_elapsed_time = time.time() - overall_start_time
        self.results['metadata']['total_evaluation_time'] = overall_elapsed_time
        
        self.log(f"\n{'#'*80}")
        self.log(f"所有级别评估完成！总耗时: {overall_elapsed_time:.2f}秒")
        self.log(f"{'#'*80}\n")
    
    def generate_summary(self):
        """生成评估总结"""
        self.log("\n生成评估总结...")
        
        summary = {
            'evaluation_metadata': self.results['metadata'],
            'levels_evaluated': self.levels,
            'metrics_summary': {},
            'detailed_results': {}
        }
        
        # 按指标汇总所有级别的结果
        metrics_list = ['tlpips', 'warping_error', 'fid', 'fvd', 'edge_iou']
        
        for metric in metrics_list:
            summary['metrics_summary'][metric] = {}
            
            for level in self.levels:
                level_data = self.results['results_by_level'].get(level, {})
                metrics = level_data.get('metrics', {})
                
                if metric in metrics:
                    metric_data = metrics[metric]
                    
                    # 提取关键指标值
                    if metric == 'tlpips':
                        value = metric_data.get('average_tlpips') or metric_data.get('mean_tlpips')
                    elif metric == 'warping_error':
                        value = metric_data.get('average_warping_error') or metric_data.get('mean_warping_error')
                    elif metric == 'fid':
                        value = metric_data.get('fid_score') or metric_data.get('fid')
                    elif metric == 'fvd':
                        value = metric_data.get('fvd_score') or metric_data.get('fvd')
                    elif metric == 'edge_iou':
                        value = metric_data.get('average_edge_iou') or metric_data.get('mean_edge_iou')
                    else:
                        value = None
                    
                    summary['metrics_summary'][metric][level] = {
                        'value': value,
                        'full_data': metric_data
                    }
        
        # 添加详细结果
        summary['detailed_results'] = self.results['results_by_level']
        
        # 保存总结
        summary_file = os.path.join(self.summary_dir, 'complete_evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"评估总结保存到: {summary_file}")
        
        # 生成人类可读的报告
        self.generate_readable_report(summary)
        
        return summary
    
    def generate_readable_report(self, summary):
        """生成人类可读的报告"""
        report_file = os.path.join(self.summary_dir, 'evaluation_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("视频质量评估完整报告\n")
            f.write("="*80 + "\n\n")
            
            # 元数据
            f.write("评估信息\n")
            f.write("-"*80 + "\n")
            metadata = summary['evaluation_metadata']
            f.write(f"评估时间: {metadata['timestamp']}\n")
            f.write(f"生成视频目录: {metadata['generated_videos_dir']}\n")
            f.write(f"真实视频目录: {metadata['real_videos_dir']}\n")
            f.write(f"设备: {metadata['device']}\n")
            f.write(f"总耗时: {metadata.get('total_evaluation_time', 0):.2f} 秒\n")
            f.write(f"评估级别: {', '.join(metadata['levels'])}\n\n")
            
            # 指标总结表格
            f.write("指标总结\n")
            f.write("="*80 + "\n\n")
            
            metrics_info = {
                'tlpips': {'name': 'Temporal LPIPS (tLPIPS)', 'lower_better': True},
                'warping_error': {'name': 'Warping Error', 'lower_better': True},
                'fid': {'name': 'Fréchet Inception Distance (FID)', 'lower_better': True},
                'fvd': {'name': 'Fréchet Video Distance (FVD)', 'lower_better': True},
                'edge_iou': {'name': 'Edge Consistency IoU', 'lower_better': False}
            }
            
            # 创建表格
            f.write(f"{'指标':<40} ")
            for level in self.levels:
                f.write(f"{level:<15} ")
            f.write("\n")
            f.write("-"*80 + "\n")
            
            for metric, metric_data in summary['metrics_summary'].items():
                info = metrics_info.get(metric, {'name': metric})
                f.write(f"{info['name']:<40} ")
                
                for level in self.levels:
                    level_data = metric_data.get(level, {})
                    value = level_data.get('value')
                    
                    if value is not None:
                        if isinstance(value, float):
                            f.write(f"{value:<15.6f} ")
                        else:
                            f.write(f"{str(value):<15} ")
                    else:
                        f.write(f"{'N/A':<15} ")
                
                f.write("\n")
            
            f.write("\n")
            
            # 详细结果
            f.write("\n" + "="*80 + "\n")
            f.write("详细评估结果\n")
            f.write("="*80 + "\n\n")
            
            for level in self.levels:
                f.write(f"\n级别: {level}\n")
                f.write("-"*80 + "\n")
                
                level_results = summary['detailed_results'].get(level, {})
                success_info = level_results.get('success', {})
                metrics = level_results.get('metrics', {})
                errors = level_results.get('errors', [])
                
                f.write("评估状态:\n")
                for metric_name, success in success_info.items():
                    status = "✓ 成功" if success else "✗ 失败"
                    f.write(f"  {metric_name}: {status}\n")
                
                if errors:
                    f.write("\n错误信息:\n")
                    for error in errors:
                        f.write(f"  - {error}\n")
                
                f.write("\n指标详情:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"  {metric_name.upper()}:\n")
                    
                    # 提取并显示关键信息
                    for key, val in metric_value.items():
                        if isinstance(val, (int, float)):
                            f.write(f"    {key}: {val:.6f}\n")
                        elif isinstance(val, str):
                            f.write(f"    {key}: {val}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("报告生成完成\n")
            f.write("="*80 + "\n")
        
        self.log(f"人类可读报告保存到: {report_file}")
    
    def run(self):
        """运行完整评估流程"""
        try:
            # 1. 评估所有级别
            self.evaluate_all_levels()
            
            # 2. 生成总结
            summary = self.generate_summary()
            
            self.log("\n" + "="*80)
            self.log("评估流程全部完成！")
            self.log("="*80)
            
            return True
            
        except Exception as e:
            self.log(f"评估流程发生错误: {str(e)}", 'ERROR')
            traceback.print_exc()
            return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='完整的多级别视频质量评估系统 - 修复版',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--videos_dir', type=str, required=True, help='生成的视频目录路径')
    parser.add_argument('--real_videos_dir', type=str, required=True, help='真实视频目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='评估结果输出目录')
    parser.add_argument('--levels', type=str, nargs='+', default=['level_0', 'level_1', 'level_2', 'level_3'], help='要评估的级别列表')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='运行设备')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已存在的评估结果')
    parser.add_argument('--temporal_checkpoint', type=str, default='', help='Temporal模型checkpoint路径（用于Warping Error评估）')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 验证输入目录
    if not os.path.exists(args.videos_dir):
        print(f"错误: 生成视频目录不存在: {args.videos_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.real_videos_dir):
        print(f"错误: 真实视频目录不存在: {args.real_videos_dir}")
        sys.exit(1)
    
    # 创建评估流水线
    pipeline = VideoEvaluationPipeline(args)
    
    # 运行评估
    success = pipeline.run()
    
    if success:
        print("\n" + "="*80)
        print("评估成功完成！")
        print(f"结果保存在: {args.output_dir}")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("评估过程中出现错误，请检查日志")
        print("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()