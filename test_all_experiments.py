# =========================
# Комплексное тестирование всех экспериментов
# Проверка результатов на валидность и выявление аномалий
# =========================

import os
import sys
import subprocess
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
import numpy as np

@dataclass
class TestResult:
    """Результат теста одного эксперимента"""
    project_name: str
    status: str  # "passed", "failed", "warning", "error"
    expected_values: Dict[str, float]
    actual_values: Dict[str, float]
    differences: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    execution_time: float
    log_file: Optional[str] = None

class ExperimentTester:
    """Класс для тестирования всех экспериментов"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.results: List[TestResult] = []
        self.expected_results = {
            "01-active-sleep-mnist": {
                "task_a_after_sleep": 96.30,
                "task_b_after_sleep": 84.12,
                "backbone_probe": 91.58,
            },
            "02-temporal-lora-gpt2": {
                "router_accuracy": 100.0,
                "shakespeare_routing": 97.2,
                "python_routing": 99.5,
            },
            "03-stability-first-basic": {
                "task_a_before_b": 99.12,
                "task_b_after_b_baseline": 98.56,
                "task_a_after_b_baseline": 0.00,
                "task_b_after_b_stability": 82.02,
                "task_a_after_b_stability": 93.52,
            },
            "04-stability-first-reversibility": {
                "task_a_before_b": 99.12,
                "task_b_after_b_baseline": 98.11,
                "task_a_after_b_baseline": 0.00,
                "task_b_after_b_stability": 84.32,
                "task_a_after_b_stability": 94.65,
            },
            "05-recursive-time-full-suite": {
                "task_a_before_b": 99.12,
                "task_a_after_b_baseline": 0.00,
                "task_a_after_b_fixed": 94.65,
                "task_a_after_b_fractal": 94.71,
                "task_a_after_b_adaptive": 94.34,
                "task_a_after_b_dream": 94.24,
            },
            "06-subjective-time-critic": {
                "lambda_start": 1805.0,
                "lambda_end": 2647.0,
            },
            "07-stability-first-cifar10": {
                "task_a_before_b": 85.0,  # Ожидаемая точность для CIFAR-10
                "task_b_after_b_baseline": 85.0,
                "task_a_after_b_baseline": 0.00,
                "task_b_after_b_stability": 80.0,
                "task_a_after_b_stability": 75.0,  # Сохранение знаний на CIFAR-10
            },
            "08-stability-first-imagenet": {
                "task_a_before_b": 65.0,  # Для CIFAR-100 или упрощенного ImageNet
                "task_b_after_b_baseline": 65.0,
                "task_a_after_b_baseline": 0.00,
                "task_b_after_b_stability": 60.0,
                "task_a_after_b_stability": 55.0,  # Сохранение знаний
            },
        }
        
        # Допустимые отклонения (в процентах)
        self.tolerance = {
            "01-active-sleep-mnist": 2.0,  # ±2%
            "02-temporal-lora-gpt2": 1.0,  # ±1% для 100% точности
            "03-stability-first-basic": 3.0,  # ±3%
            "04-stability-first-reversibility": 3.0,
            "05-recursive-time-full-suite": 3.0,
            "06-subjective-time-critic": 5.0,  # ±5% для lambda
            "07-stability-first-cifar10": 5.0,  # ±5% для CIFAR-10 (более сложный датасет)
            "08-stability-first-imagenet": 8.0,  # ±8% для ImageNet/CIFAR-100 (еще сложнее)
        }
    
    def run_experiment(self, project_dir: str, script_name: str) -> Tuple[bool, str, Dict[str, float], float]:
        """Запускает эксперимент и парсит результаты"""
        # Нормализуем путь - используем абсолютные пути
        root = Path(self.root_dir).resolve()
        project_path = root / project_dir
        script_path = project_path / script_name
        
        if not script_path.exists():
            return False, f"Script not found: {script_path}", {}, 0.0
        
        print(f"\n{'='*80}")
        print(f"Запуск: {project_dir}/{script_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Используем относительный путь к скрипту относительно project_path
            script_rel_path = script_path.relative_to(project_path)
            
            # Запускаем скрипт и перехватываем вывод
            result = subprocess.run(
                [sys.executable, str(script_rel_path)],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 минут максимум (для CIFAR-10 и ImageNet нужно больше времени)
                encoding='utf-8',
                errors='replace'
            )
            
            execution_time = time.time() - start_time
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Извлекаем имя проекта для парсинга
            project_name = project_dir.replace("\\", "/").split("/")[-1]
            
            # Парсим результаты из вывода
            parsed_results = self.parse_results(project_name, stdout + stderr)
            
            if result.returncode != 0:
                return False, f"Ошибка выполнения: {stderr}", parsed_results, execution_time
            
            return True, stdout, parsed_results, execution_time
            
        except subprocess.TimeoutExpired:
            return False, "Превышено время ожидания (10 минут)", {}, time.time() - start_time
        except Exception as e:
            return False, f"Исключение: {str(e)}", {}, time.time() - start_time
    
    def parse_results(self, project_name: str, output: str) -> Dict[str, float]:
        """Парсит числовые результаты из вывода"""
        results = {}
        
        # Универсальный паттерн для поиска процентов
        percent_pattern = r"(\d+\.\d+)%"
        
        if project_name == "01-active-sleep-mnist":
            # Ищем результаты в формате [title] acc=XX.XX%
            matches = re.findall(r"\[.*?\]\s*acc=(\d+\.\d+)%", output)
            if len(matches) >= 3:
                results["task_a_after_sleep"] = float(matches[0]) if len(matches) > 0 else 0.0
                results["task_b_after_sleep"] = float(matches[1]) if len(matches) > 1 else 0.0
                results["backbone_probe"] = float(matches[2]) if len(matches) > 2 else 0.0
        
        elif project_name == "02-temporal-lora-gpt2":
            # Ищем Router Accuracy
            router_match = re.search(r"Router\s+Accuracy[:\s]+(\d+\.\d+)%", output, re.IGNORECASE)
            if router_match:
                results["router_accuracy"] = float(router_match.group(1))
            
            # Ищем routing результаты
            shakespeare_match = re.search(r"Shakespeare.*?(\d+\.\d+)%", output, re.IGNORECASE)
            if shakespeare_match:
                results["shakespeare_routing"] = float(shakespeare_match.group(1))
            
            python_match = re.search(r"Python.*?(\d+\.\d+)%", output, re.IGNORECASE)
            if python_match:
                results["python_routing"] = float(python_match.group(1))
        
        elif project_name in ["03-stability-first-basic", "04-stability-first-reversibility"]:
            # Парсим таблицу результатов
            # Формат: Task A (0-4) before B | XX.XX% | XX.XX%
            lines = output.split('\n')
            for line in lines:
                if "Task A (0-4) before B" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_before_b"] = float(matches[0])
                
                elif "Task B (5-9) after B" in line:
                    matches = re.findall(percent_pattern, line)
                    if len(matches) >= 2:
                        results["task_b_after_b_baseline"] = float(matches[0])
                        results["task_b_after_b_stability"] = float(matches[1])
                
                elif "Task A (0-4) after B" in line:
                    matches = re.findall(percent_pattern, line)
                    if len(matches) >= 2:
                        results["task_a_after_b_baseline"] = float(matches[0])
                        results["task_a_after_b_stability"] = float(matches[1])
            
            # Альтернативный способ: ищем в формате [title] acc=XX.XX%
            if not results:
                acc_matches = re.findall(r"\[.*?\]\s*acc=(\d+\.\d+)%", output)
                if len(acc_matches) >= 5:
                    results["task_a_before_b"] = float(acc_matches[0])
                    results["task_b_after_b_baseline"] = float(acc_matches[1])
                    results["task_a_after_b_baseline"] = float(acc_matches[2])
                    results["task_b_after_b_stability"] = float(acc_matches[3])
                    results["task_a_after_b_stability"] = float(acc_matches[4])
        
        elif project_name == "05-recursive-time-full-suite":
            # Парсим таблицу результатов
            lines = output.split('\n')
            for line in lines:
                if "Task A (0-4) before B" in line or "A before" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_before_b"] = float(matches[0])
                
                # Ищем результаты для разных методов
                if "baseline" in line.lower() and "A after" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_after_b_baseline"] = float(matches[-1])
                
                if "fixed" in line.lower() and "A after" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_after_b_fixed"] = float(matches[-1])
                
                if "fractal" in line.lower() and "A after" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_after_b_fractal"] = float(matches[-1])
                
                if "adaptive" in line.lower() and "A after" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_after_b_adaptive"] = float(matches[-1])
                
                if "dream" in line.lower() and "A after" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_after_b_dream"] = float(matches[-1])
        
        elif project_name == "06-subjective-time-critic":
            # Ищем все значения Lambda
            lambda_matches = re.findall(r"Lambda[:\s]+(\d+\.\d+)", output, re.IGNORECASE)
            if len(lambda_matches) >= 2:
                values = [float(m) for m in lambda_matches]
                results["lambda_start"] = values[0]
                results["lambda_end"] = values[-1]
            elif len(lambda_matches) == 1:
                results["lambda_start"] = float(lambda_matches[0])
                results["lambda_end"] = float(lambda_matches[0])
        
        elif project_name in ["07-stability-first-cifar10", "08-stability-first-imagenet"]:
            # Парсим таблицу результатов (аналогично 03, 04)
            lines = output.split('\n')
            for line in lines:
                if "Task A" in line and "before B" in line:
                    matches = re.findall(percent_pattern, line)
                    if matches:
                        results["task_a_before_b"] = float(matches[0])
                
                elif "Task B" in line and "after B" in line:
                    matches = re.findall(percent_pattern, line)
                    if len(matches) >= 2:
                        results["task_b_after_b_baseline"] = float(matches[0])
                        results["task_b_after_b_stability"] = float(matches[1])
                    elif len(matches) == 1:
                        # Может быть только baseline или только stability
                        if "baseline" in line.lower():
                            results["task_b_after_b_baseline"] = float(matches[0])
                        elif "stability" in line.lower():
                            results["task_b_after_b_stability"] = float(matches[0])
                
                elif "Task A" in line and "after B" in line:
                    matches = re.findall(percent_pattern, line)
                    if len(matches) >= 2:
                        results["task_a_after_b_baseline"] = float(matches[0])
                        results["task_a_after_b_stability"] = float(matches[1])
                    elif len(matches) == 1:
                        if "baseline" in line.lower():
                            results["task_a_after_b_baseline"] = float(matches[0])
                        elif "stability" in line.lower():
                            results["task_a_after_b_stability"] = float(matches[0])
            
            # Альтернативный способ: ищем в формате [title] acc=XX.XX%
            if not results:
                acc_matches = re.findall(r"\[.*?\]\s*acc=(\d+\.\d+)%", output)
                if len(acc_matches) >= 5:
                    results["task_a_before_b"] = float(acc_matches[0])
                    results["task_b_after_b_baseline"] = float(acc_matches[1])
                    results["task_a_after_b_baseline"] = float(acc_matches[2])
                    results["task_b_after_b_stability"] = float(acc_matches[3])
                    results["task_a_after_b_stability"] = float(acc_matches[4])
        
        return results
    
    def validate_results(self, project_name: str, expected: Dict[str, float], 
                        actual: Dict[str, float]) -> Tuple[str, List[str], List[str]]:
        """Валидирует результаты и возвращает статус, предупреждения и ошибки"""
        warnings = []
        errors = []
        status = "passed"
        
        tolerance = self.tolerance.get(project_name, 5.0)
        
        # Проверяем каждое ожидаемое значение
        for key, expected_value in expected.items():
            if key not in actual:
                warnings.append(f"Не найдено значение для {key}")
                continue
            
            actual_value = actual[key]
            difference = abs(actual_value - expected_value)
            relative_diff = (difference / expected_value * 100) if expected_value > 0 else difference
            
            # Специальная проверка для baseline забывания Task A (должен быть близок к 0%)
            if "task_a_after_b_baseline" in key:
                if actual_value > 5.0:  # Если baseline показывает >5%, это подозрительно
                    errors.append(
                        f"Baseline забывание Task A слишком слабое: {actual_value:.2f}% "
                        f"(ожидалось ~0%, разница: {difference:.2f}%)"
                    )
                    status = "failed"
            
            # Проверка на слишком хорошие результаты (возможное переобучение)
            if actual_value > 99.5 and "accuracy" not in key.lower():
                warnings.append(
                    f"Подозрительно высокий результат для {key}: {actual_value:.2f}% "
                    f"(возможно переобучение)"
                )
            
            # Проверка на соответствие ожидаемым значениям
            if relative_diff > tolerance:
                if "baseline" in key and expected_value == 0.0:
                    # Для baseline 0% более строгая проверка
                    if actual_value > 1.0:
                        errors.append(
                            f"Критическое отклонение для {key}: {actual_value:.2f}% "
                            f"(ожидалось {expected_value:.2f}%, разница: {difference:.2f}%)"
                        )
                        status = "failed"
                else:
                    warnings.append(
                        f"Отклонение для {key}: {actual_value:.2f}% "
                        f"(ожидалось {expected_value:.2f}%, разница: {difference:.2f}%)"
                    )
                    if relative_diff > tolerance * 2:
                        status = "warning"
        
        # Проверка на отсутствие критических метрик
        critical_metrics = {
            "03-stability-first-basic": ["task_a_after_b_stability"],
            "04-stability-first-reversibility": ["task_a_after_b_stability"],
            "05-recursive-time-full-suite": ["task_a_after_b_fixed"],
            "07-stability-first-cifar10": ["task_a_after_b_stability"],
            "08-stability-first-imagenet": ["task_a_after_b_stability"],
        }
        
        if project_name in critical_metrics:
            for metric in critical_metrics[project_name]:
                if metric not in actual:
                    errors.append(f"Отсутствует критическая метрика: {metric}")
                    status = "failed"
        
        return status, warnings, errors
    
    def test_project(self, project_dir: str, script_name: str) -> TestResult:
        """Тестирует один проект"""
        project_name = project_dir.replace("\\", "/").split("/")[-1]
        
        success, output, actual_results, exec_time = self.run_experiment(project_dir, script_name)
        
        expected = self.expected_results.get(project_name, {})
        
        if not success:
            return TestResult(
                project_name=project_name,
                status="error",
                expected_values=expected,
                actual_values=actual_results,
                differences={},
                warnings=[],
                errors=[output],
                execution_time=exec_time,
            )
        
        # Вычисляем разницы
        differences = {}
        for key in expected:
            if key in actual_results:
                differences[key] = actual_results[key] - expected[key]
        
        # Валидируем результаты
        status, warnings, errors = self.validate_results(
            project_name, expected, actual_results
        )
        
        return TestResult(
            project_name=project_name,
            status=status,
            expected_values=expected,
            actual_values=actual_results,
            differences=differences,
            warnings=warnings,
            errors=errors,
            execution_time=exec_time,
        )
    
    def run_all_tests(self) -> Dict[str, any]:
        """Запускает все тесты"""
        test_configs = [
            ("01-active-sleep-mnist", "active_sleep.py"),
            ("02-temporal-lora-gpt2", "temporal_lora.py"),
            ("03-stability-first-basic", "run_demo.py"),
            ("04-stability-first-reversibility", "run_demo.py"),
            ("05-recursive-time-full-suite", "run_split_suite.py"),
            ("06-subjective-time-critic", "demo_6_subjective_time.py"),
            ("07-stability-first-cifar10", "run_demo.py"),
            ("08-stability-first-imagenet", "run_demo.py"),
        ]
        
        print("\n" + "="*80)
        print("НАЧАЛО КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ ВСЕХ ЭКСПЕРИМЕНТОВ")
        print("="*80)
        
        for project_dir, script_name in test_configs:
            result = self.test_project(project_dir, script_name)
            self.results.append(result)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, any]:
        """Генерирует итоговый отчет"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        warnings_count = sum(1 for r in self.results if r.status == "warning")
        errors_count = sum(1 for r in self.results if r.status == "error")
        
        report = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings_count,
                "errors": errors_count,
            },
            "results": [asdict(r) for r in self.results],
        }
        
        # Выводим отчет
        print("\n" + "="*80)
        print("ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
        print("="*80)
        print(f"\nВсего экспериментов: {total}")
        print(f"[OK] Успешно: {passed}")
        print(f"[!] Предупреждения: {warnings_count}")
        print(f"[X] Провалено: {failed}")
        print(f"[ERROR] Ошибки выполнения: {errors_count}")
        
        print("\n" + "-"*80)
        print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print("-"*80)
        
        for result in self.results:
            status_icon = {
                "passed": "[OK]",
                "warning": "[!]",
                "failed": "[X]",
                "error": "[ERROR]",
            }.get(result.status, "[?]")
            
            print(f"\n{status_icon} {result.project_name}")
            print(f"   Статус: {result.status}")
            
            if result.errors:
                print(f"   Ошибки:")
                for error in result.errors:
                    print(f"     - {error}")
            
            if result.warnings:
                print(f"   Предупреждения:")
                for warning in result.warnings:
                    print(f"     - {warning}")
            
            if result.actual_values:
                print(f"   Полученные результаты:")
                for key, value in result.actual_values.items():
                    expected = result.expected_values.get(key, "N/A")
                    diff = result.differences.get(key, 0)
                    diff_str = f" ({diff:+.2f})" if diff != 0 else ""
                    print(f"     {key}: {value:.2f} (ожидалось: {expected}){diff_str}")
        
        # Сохраняем отчет в JSON
        report_path = self.root_dir / "logs" / "test_report.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] Полный отчет сохранен в: {report_path}")
        
        return report

def main():
    """Главная функция"""
    tester = ExperimentTester(root_dir=".")
    report = tester.run_all_tests()
    
    # Возвращаем код выхода
    if report["summary"]["failed"] > 0 or report["summary"]["errors"] > 0:
        sys.exit(1)
    elif report["summary"]["warnings"] > 0:
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

