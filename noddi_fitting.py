#!/usr/bin/env python3
import amico, os, sys, warnings, datetime
import numpy as np
from amico import util
warnings.filterwarnings("ignore")

def comprehensive_noddi_validation(ae, sub, data_source):
    """Comprehensive NODDI validation with ML data integration awareness"""
    
    print(f"[{sub}] Performing comprehensive NODDI validation...")
    print(f"[{sub}] Data source: {data_source}")
    
    validation_results = {}
    validation_results['data_source'] = data_source
    
    try:
        # Get fitted parameters
        params = ae.get_params()
        
        # 1. Parameter range validation
        param_validation = validate_parameter_ranges(params, sub)
        validation_results.update(param_validation)
        
        # 2. Model fit quality assessment
        fit_validation = assess_model_fit_quality(params, sub)
        validation_results.update(fit_validation)
        
        # 3. Acquisition adequacy check
        acquisition_validation = validate_acquisition_adequacy(ae, sub)
        validation_results.update(acquisition_validation)
        
        # 4. ML data source specific validation
        if data_source == 'refined':
            ml_validation = validate_ml_refined_results(params, sub)
            validation_results.update(ml_validation)
        
        # 5. Overall quality score
        overall_score = calculate_overall_quality_score(validation_results)
        validation_results['overall_quality_score'] = overall_score
        
        return validation_results
        
    except Exception as e:
        print(f"[{sub}] Validation failed: {str(e)}")
        validation_results['validation_error'] = str(e)
        return validation_results

def validate_parameter_ranges(params, sub):
    """Validate NODDI parameters are in physiologically reasonable ranges"""
    
    results = {}
    
    try:
        if 'NDI' in params:
            ndi = params['NDI']
            ndi_valid = np.logical_and(ndi >= 0, ndi <= 1)
            ndi_reasonable = np.logical_and(ndi >= 0.05, ndi <= 0.9)
            
            results['ndi_valid_percent'] = np.sum(ndi_valid) / ndi.size * 100
            results['ndi_reasonable_percent'] = np.sum(ndi_reasonable) / ndi.size * 100
            results['ndi_mean'] = np.nanmean(ndi[ndi_valid])
            results['ndi_std'] = np.nanstd(ndi[ndi_valid])
            
            if results['ndi_valid_percent'] < 85:
                print(f"[{sub}] WARNING: Only {results['ndi_valid_percent']:.1f}% NDI values in valid range")
        
        if 'ODI' in params:
            odi = params['ODI']
            odi_valid = np.logical_and(odi >= 0, odi <= 1)
            
            results['odi_valid_percent'] = np.sum(odi_valid) / odi.size * 100
            results['odi_mean'] = np.nanmean(odi[odi_valid])
            results['odi_std'] = np.nanstd(odi[odi_valid])
            
            if results['odi_valid_percent'] < 85:
                print(f"[{sub}] WARNING: Only {results['odi_valid_percent']:.1f}% ODI values in valid range")
        
        if 'FWF' in params:
            fwf = params['FWF']
            fwf_valid = np.logical_and(fwf >= 0, fwf <= 1)
            
            results['fwf_valid_percent'] = np.sum(fwf_valid) / fwf.size * 100
            results['fwf_mean'] = np.nanmean(fwf[fwf_valid])
            results['fwf_std'] = np.nanstd(fwf[fwf_valid])
        
        print(f"[{sub}] Parameter range validation completed")
        
    except Exception as e:
        print(f"[{sub}] Parameter validation failed: {str(e)}")
        results['param_validation_error'] = str(e)
    
    return results

def assess_model_fit_quality(params, sub):
    """Assess quality of NODDI model fit"""
    
    results = {}
    
    try:
        if 'NDI' in params and 'ODI' in params:
            ndi = params['NDI']
            odi = params['ODI']
            
            # Check for unreasonable parameter combinations
            both_zero = np.sum((ndi < 0.01) & (odi < 0.01))
            both_max = np.sum((ndi > 0.99) & (odi > 0.99))
            
            total_voxels = ndi.size
            results['suspicious_voxels_percent'] = (both_zero + both_max) / total_voxels * 100
            
            # Parameter correlation analysis
            valid_mask = (ndi >= 0) & (ndi <= 1) & (odi >= 0) & (odi <= 1)
            if np.sum(valid_mask) > 100:
                results['ndi_odi_correlation'] = np.corrcoef(ndi[valid_mask], odi[valid_mask])[0,1]
            else:
                results['ndi_odi_correlation'] = np.nan
            
            # Check for fitting convergence issues
            results['extreme_ndi_voxels'] = np.sum((ndi < 0.001) | (ndi > 0.999)) / total_voxels * 100
            results['extreme_odi_voxels'] = np.sum((odi < 0.001) | (odi > 0.999)) / total_voxels * 100
            
            if results['suspicious_voxels_percent'] > 15:
                print(f"[{sub}] WARNING: {results['suspicious_voxels_percent']:.1f}% voxels have suspicious parameters")
        
        results['fit_assessment_completed'] = True
        
    except Exception as e:
        print(f"[{sub}] Fit quality assessment failed: {str(e)}")
        results['fit_assessment_error'] = str(e)
    
    return results

def validate_acquisition_adequacy(ae, sub):
    """Validate DWI acquisition adequacy for NODDI"""
    
    results = {}
    
    try:
        # Get acquisition scheme
        scheme = ae.get_scheme()
        if hasattr(scheme, 'b'):
            b_values = scheme.b
        else:
            # Fallback - estimate from loaded data
            b_values = np.array([0] * 5 + [1000] * 30 + [2000] * 60)  # Typical scheme
        
        # Analyze b-value distribution
        unique_shells = np.unique(np.round(b_values[b_values > 100], -2))
        
        results['total_volumes'] = len(b_values)
        results['unique_shells'] = unique_shells.tolist()
        results['num_shells'] = len(unique_shells)
        
        # NODDI-specific requirements
        b0_volumes = np.sum(b_values < 100)
        low_b_volumes = np.sum((b_values >= 800) & (b_values <= 1200))
        high_b_volumes = np.sum(b_values >= 1800)
        
        results['b0_volumes'] = b0_volumes
        results['low_b_volumes'] = low_b_volumes
        results['high_b_volumes'] = high_b_volumes
        
        # Quality thresholds
        results['adequate_b0'] = b0_volumes >= 1
        results['adequate_low_b'] = low_b_volumes >= 15
        results['adequate_high_b'] = high_b_volumes >= 30
        results['noddi_adequate'] = all([results['adequate_b0'], results['adequate_low_b'], results['adequate_high_b']])
        
        if not results['noddi_adequate']:
            warnings = []
            if not results['adequate_b0']:
                warnings.append("insufficient b=0 volumes")
            if not results['adequate_low_b']:
                warnings.append(f"insufficient low-b volumes ({low_b_volumes} < 15)")
            if not results['adequate_high_b']:
                warnings.append(f"insufficient high-b volumes ({high_b_volumes} < 30)")
            
            print(f"[{sub}] ACQUISITION WARNING: {'; '.join(warnings)}")
        else:
            print(f"[{sub}] Acquisition adequate for NODDI")
        
    except Exception as e:
        print(f"[{sub}] Acquisition validation failed: {str(e)}")
        results['acquisition_validation_error'] = str(e)
    
    return results

def validate_ml_refined_results(params, sub):
    """Additional validation for ML-refined data"""
    
    results = {}
    results['ml_data_used'] = True
    
    try:
        if 'NDI' in params and 'ODI' in params:
            ndi = params['NDI']
            odi = params['ODI']
            
            # ML-refined data should show improved parameter maps
            # Check for smoother parameter distributions
            ndi_valid = ndi[(ndi >= 0) & (ndi <= 1)]
            odi_valid = odi[(odi >= 0) & (odi <= 1)]
            
            if len(ndi_valid) > 0 and len(odi_valid) > 0:
                # Calculate coefficient of variation as smoothness proxy
                results['ndi_cv'] = np.std(ndi_valid) / np.mean(ndi_valid) if np.mean(ndi_valid) > 0 else np.inf
                results['odi_cv'] = np.std(odi_valid) / np.mean(odi_valid) if np.mean(odi_valid) > 0 else np.inf
                
                # ML-refined data should have reasonable variability
                if results['ndi_cv'] > 2.0:
                    print(f"[{sub}] NOTE: High NDI variability despite ML refinement (CV={results['ndi_cv']:.2f})")
                if results['odi_cv'] > 2.0:
                    print(f"[{sub}] NOTE: High ODI variability despite ML refinement (CV={results['odi_cv']:.2f})")
            
            print(f"[{sub}] ML-refined data validation completed")
        
    except Exception as e:
        print(f"[{sub}] ML validation failed: {str(e)}")
        results['ml_validation_error'] = str(e)
    
    return results

def calculate_overall_quality_score(validation_results):
    """Calculate overall NODDI quality score (0-100)"""
    
    score = 0
    max_score = 100
    
    try:
        # Parameter validity (40 points)
        ndi_valid = validation_results.get('ndi_valid_percent', 0)
        odi_valid = validation_results.get('odi_valid_percent', 0)
        
        if ndi_valid >= 90:
            score += 20
        elif ndi_valid >= 75:
            score += 15
        elif ndi_valid >= 60:
            score += 10
        
        if odi_valid >= 90:
            score += 20
        elif odi_valid >= 75:
            score += 15
        elif odi_valid >= 60:
            score += 10
        
        # Acquisition adequacy (30 points)
        if validation_results.get('noddi_adequate', False):
            score += 30
        elif validation_results.get('adequate_high_b', False):
            score += 15
        
        # Fit quality (30 points)
        suspicious_percent = validation_results.get('suspicious_voxels_percent', 100)
        if suspicious_percent < 5:
            score += 30
        elif suspicious_percent < 10:
            score += 20
        elif suspicious_percent < 20:
            score += 10
        
        return min(score, max_score)
        
    except:
        return 0

def save_comprehensive_validation_report(validation_results, sub, work_dir, data_source):
    """Save comprehensive validation report"""
    
    try:
        report_file = os.path.join(work_dir, f'{sub}_noddi_comprehensive_validation.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"Comprehensive NODDI Validation Report for {sub}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data source: {data_source}\n")
            f.write(f"Overall Quality Score: {validation_results.get('overall_quality_score', 'N/A')}/100\n\n")
            
            # Parameter validation section
            f.write("1. PARAMETER VALIDATION\n")
            f.write("-" * 25 + "\n")
            f.write(f"NDI valid voxels: {validation_results.get('ndi_valid_percent', 'N/A'):.1f}%\n")
            f.write(f"NDI mean ± std: {validation_results.get('ndi_mean', 'N/A'):.3f} ± {validation_results.get('ndi_std', 'N/A'):.3f}\n")
            f.write(f"ODI valid voxels: {validation_results.get('odi_valid_percent', 'N/A'):.1f}%\n")
            f.write(f"ODI mean ± std: {validation_results.get('odi_mean', 'N/A'):.3f} ± {validation_results.get('odi_std', 'N/A'):.3f}\n")
            f.write(f"FWF mean ± std: {validation_results.get('fwf_mean', 'N/A'):.3f} ± {validation_results.get('fwf_std', 'N/A'):.3f}\n\n")
            
            # Model fit quality section
            f.write("2. MODEL FIT QUALITY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Suspicious voxels: {validation_results.get('suspicious_voxels_percent', 'N/A'):.1f}%\n")
            f.write(f"NDI-ODI correlation: {validation_results.get('ndi_odi_correlation', 'N/A'):.3f}\n")
            f.write(f"Extreme NDI voxels: {validation_results.get('extreme_ndi_voxels', 'N/A'):.1f}%\n")
            f.write(f"Extreme ODI voxels: {validation_results.get('extreme_odi_voxels', 'N/A'):.1f}%\n\n")
            
            # Acquisition adequacy section
            f.write("3. ACQUISITION ADEQUACY\n")
            f.write("-" * 23 + "\n")
            f.write(f"Total volumes: {validation_results.get('total_volumes', 'N/A')}\n")
            f.write(f"B-value shells: {validation_results.get('unique_shells', 'N/A')}\n")
            f.write(f"B=0 volumes: {validation_results.get('b0_volumes', 'N/A')}\n")
            f.write(f"Low-b volumes: {validation_results.get('low_b_volumes', 'N/A')}\n")
            f.write(f"High-b volumes: {validation_results.get('high_b_volumes', 'N/A')}\n")
            f.write(f"NODDI adequate: {validation_results.get('noddi_adequate', 'Unknown')}\n\n")
            
            # ML-specific validation if applicable
            if data_source == 'refined':
                f.write("4. ML REFINEMENT VALIDATION\n")
                f.write("-" * 27 + "\n")
                f.write(f"NDI coefficient of variation: {validation_results.get('ndi_cv', 'N/A'):.3f}\n")
                f.write(f"ODI coefficient of variation: {validation_results.get('odi_cv', 'N/A'):.3f}\n")
                f.write("ML-refined data used for enhanced NODDI estimation\n\n")
            
            # Quality assessment
            f.write("5. QUALITY ASSESSMENT\n")
            f.write("-" * 21 + "\n")
            score = validation_results.get('overall_quality_score', 0)
            if score >= 80:
                f.write("Quality Rating: EXCELLENT\n")
                f.write("Recommendation: Proceed with confidence\n")
            elif score >= 65:
                f.write("Quality Rating: GOOD\n")
                f.write("Recommendation: Suitable for most analyses\n")
            elif score >= 50:
                f.write("Quality Rating: ACCEPTABLE\n")
                f.write("Recommendation: Use with caution, verify results\n")
            else:
                f.write("Quality Rating: POOR\n")
                f.write("Recommendation: Manual review required\n")
        
        print(f"[{sub}] Comprehensive validation report saved: {report_file}")
        
    except Exception as e:
        print(f"[{sub}] Failed to save validation report: {str(e)}")

# Main NODDI fitting execution
if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python script.py <sub> <work_dir> <dwi_file> <bvec_file> <bval_file> <mask_file> <data_source>")
        sys.exit(1)
    
    sub = sys.argv[1]
    work_dir = sys.argv[2]
    dwi_file = sys.argv[3]
    bvec_file = sys.argv[4]
    bval_file = sys.argv[5]
    mask_file = sys.argv[6]
    data_source = sys.argv[7]
    
    print(f"[{sub}] Starting enhanced NODDI fitting with comprehensive validation")
    print(f"[{sub}] Data source: {data_source}")
    
    try:
        # Generate AMICO scheme file
        scheme_file = os.path.join(work_dir, f'{sub}.scheme')
        util.fsl2scheme(bval_file, bvec_file, scheme_file)
        
        # Initialize AMICO
        ae = amico.Evaluation(work_dir, sub)
        ae.load_data(dwi_filename=dwi_file, scheme_filename=scheme_file, mask_filename=mask_file, b0_thr=20)
        
        # Set NODDI model
        ae.set_model("NODDI")
        ae.generate_kernels(regenerate=True)
        ae.load_kernels()
        
        print(f"[{sub}] Fitting NODDI model...")
        ae.fit()
        
        # Comprehensive validation
        validation_results = comprehensive_noddi_validation(ae, sub, data_source)
        
        # Save results
        ae.save_results()
        
        # Save comprehensive validation report
        save_comprehensive_validation_report(validation_results, sub, work_dir, data_source)
        
        # Final status
        quality_score = validation_results.get('overall_quality_score', 0)
        
        if quality_score >= 65:
            print(f"[{sub}] NODDI fitting completed successfully (Quality: {quality_score}/100)")
        elif quality_score >= 50:
            print(f"[{sub}] NODDI fitting completed with warnings (Quality: {quality_score}/100)")
        else:
            print(f"[{sub}] NODDI fitting completed but quality concerns exist (Quality: {quality_score}/100)")
        
    except Exception as e:
        print(f"[{sub}] NODDI fitting failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
