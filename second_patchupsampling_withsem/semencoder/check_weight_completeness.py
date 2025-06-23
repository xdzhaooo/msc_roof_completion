#!/usr/bin/env python3
"""
检查权重加载的完整性 - 确保所有参数都被正确加载
"""

import torch
from model import EnhancedResNet50Encoder, AutoEncoder

def check_weight_completeness():
    """详细检查权重加载的完整性"""
    print("=" * 80)
    print("权重完整性检查")
    print("=" * 80)
    
    # 1. 检查原始完整模型的encoder部分
    print("\n1️⃣ 检查原始AutoEncoder的encoder部分...")
    try:
        # 加载原始完整模型检查点
        original_checkpoint = torch.load('checkpoint_epoch_15.pth', map_location='cpu')
        original_state_dict = original_checkpoint['model_state_dict']
        
        # 创建完整的AutoEncoder来对比
        full_model = AutoEncoder(input_channels=1, latent_dim=768)
        
        # 提取原始encoder权重
        original_encoder_weights = {}
        encoder_prefix = 'encoder.'
        for key, value in original_state_dict.items():
            if key.startswith(encoder_prefix):
                new_key = key[len(encoder_prefix):]
                original_encoder_weights[new_key] = value
        
        print(f"✓ 原始检查点中encoder参数数量: {len(original_encoder_weights)}")
        
        # 检查encoder模型期望的参数
        standalone_encoder = EnhancedResNet50Encoder(input_channels=1, latent_dim=768)
        expected_keys = set(standalone_encoder.state_dict().keys())
        extracted_keys = set(original_encoder_weights.keys())
        
        print(f"✓ 独立encoder期望参数数量: {len(expected_keys)}")
        print(f"✓ 提取的encoder参数数量: {len(extracted_keys)}")
        
    except Exception as e:
        print(f"❌ 加载原始检查点失败: {e}")
        return False
    
    # 2. 检查提取的权重文件
    print("\n2️⃣ 检查提取的权重文件...")
    try:
        extracted_checkpoint = torch.load('encoder_weights_checkpoint_epoch_15.pth', map_location='cpu')
        extracted_state_dict = extracted_checkpoint['encoder_state_dict']
        extracted_file_keys = set(extracted_state_dict.keys())
        
        print(f"✓ 提取文件中的参数数量: {len(extracted_file_keys)}")
        
    except Exception as e:
        print(f"❌ 加载提取的权重文件失败: {e}")
        return False
    
    # 3. 详细对比分析
    print("\n3️⃣ 详细对比分析...")
    
    # 检查缺失的参数
    missing_in_extracted = expected_keys - extracted_keys
    if missing_in_extracted:
        print(f"❌ 在提取权重中缺失的参数 ({len(missing_in_extracted)}):")
        for key in sorted(missing_in_extracted):
            print(f"   - {key}")
    else:
        print("✓ 没有缺失的参数")
    
    # 检查多余的参数
    unexpected_in_extracted = extracted_keys - expected_keys
    if unexpected_in_extracted:
        print(f"⚠️  提取权重中多余的参数 ({len(unexpected_in_extracted)}):")
        for key in sorted(unexpected_in_extracted):
            print(f"   - {key}")
    else:
        print("✓ 没有多余的参数")
    
    # 检查文件中的参数是否和提取的一致
    file_vs_extracted_diff = extracted_file_keys.symmetric_difference(extracted_keys)
    if file_vs_extracted_diff:
        print(f"❌ 文件中的参数与内存中提取的参数不一致:")
        for key in sorted(file_vs_extracted_diff):
            print(f"   - {key}")
    else:
        print("✓ 文件中的参数与提取的参数完全一致")
    
    # 4. 检查参数数值是否一致
    print("\n4️⃣ 检查参数数值一致性...")
    value_mismatches = 0
    for key in expected_keys.intersection(extracted_keys):
        if key in original_encoder_weights and key in extracted_state_dict:
            original_param = original_encoder_weights[key]
            extracted_param = extracted_state_dict[key]
            
            if not torch.equal(original_param, extracted_param):
                print(f"❌ 参数值不匹配: {key}")
                value_mismatches += 1
    
    if value_mismatches == 0:
        print("✓ 所有参数值都完全一致")
    else:
        print(f"❌ 发现 {value_mismatches} 个参数值不匹配")
    
    # 5. 实际加载测试
    print("\n5️⃣ 实际加载测试...")
    try:
        test_encoder = EnhancedResNet50Encoder(input_channels=1, latent_dim=768)
        missing_keys, unexpected_keys = test_encoder.load_state_dict(extracted_state_dict, strict=False)
        
        if missing_keys:
            print(f"❌ 加载时发现缺失的键 ({len(missing_keys)}):")
            for key in missing_keys:
                print(f"   - {key}")
        else:
            print("✓ 加载时没有缺失的键")
        
        if unexpected_keys:
            print(f"⚠️  加载时发现意外的键 ({len(unexpected_keys)}):")
            for key in unexpected_keys:
                print(f"   - {key}")
        else:
            print("✓ 加载时没有意外的键")
        
        # 检查strict=True模式
        try:
            test_encoder_strict = EnhancedResNet50Encoder(input_channels=1, latent_dim=768)
            test_encoder_strict.load_state_dict(extracted_state_dict, strict=True)
            print("✅ 严格模式加载成功 - 权重完全匹配！")
            strict_success = True
        except Exception as e:
            print(f"❌ 严格模式加载失败: {e}")
            strict_success = False
            
    except Exception as e:
        print(f"❌ 加载测试失败: {e}")
        return False
    
    # 6. 功能测试
    print("\n6️⃣ 功能完整性测试...")
    try:
        test_input = torch.randn(2, 1, 128, 128)
        
        # 使用原始完整模型的encoder
        full_model.load_state_dict(original_state_dict)
        full_model.eval()
        with torch.no_grad():
            original_output = full_model.encode(test_input)
        
        # 使用提取的独立encoder
        test_encoder.eval()
        with torch.no_grad():
            extracted_output = test_encoder(test_input)
        
        # 对比输出
        output_diff = torch.abs(original_output - extracted_output).max()
        print(f"✓ 输出差异: {output_diff.item():.2e}")
        
        if output_diff < 1e-5:
            print("✅ 输出完全一致 - 功能验证通过！")
            functional_success = True
        else:
            print("⚠️  输出存在差异，但可能在可接受范围内")
            functional_success = output_diff < 1e-3
            
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        functional_success = False
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("📊 完整性检查总结")
    print("=" * 80)
    
    completeness_score = 0
    total_checks = 5
    
    if len(missing_in_extracted) == 0:
        print("✅ 参数完整性: 通过")
        completeness_score += 1
    else:
        print("❌ 参数完整性: 失败")
    
    if value_mismatches == 0:
        print("✅ 数值一致性: 通过")
        completeness_score += 1
    else:
        print("❌ 数值一致性: 失败")
    
    if len(missing_keys) == 0:
        print("✅ 加载完整性: 通过")
        completeness_score += 1
    else:
        print("❌ 加载完整性: 失败")
    
    if strict_success:
        print("✅ 严格加载: 通过")
        completeness_score += 1
    else:
        print("❌ 严格加载: 失败")
    
    if functional_success:
        print("✅ 功能一致性: 通过")
        completeness_score += 1
    else:
        print("❌ 功能一致性: 失败")
    
    print(f"\n🎯 总体完整性评分: {completeness_score}/{total_checks} ({completeness_score/total_checks*100:.1f}%)")
    
    if completeness_score == total_checks:
        print("🎉 权重加载完整性验证通过 - 可以安全使用！")
        return True
    else:
        print("⚠️  存在一些问题，建议检查上述失败项目")
        return False


if __name__ == "__main__":
    success = check_weight_completeness()
    exit(0 if success else 1) 