class XAIAuditAgent:
    """Agent 2: Parses XAI attribute scores into human-readable engineering audit logs."""
    def compile_audit_justification(self, attributions: dict, risk_score: float) -> str:
        top_feature = max(attributions, key=attributions.get)
        impact_val = attributions[top_feature]
        
        if risk_score > 0.6:
            return f"URGENT: Maintenance mandated. Primary driver is '{top_feature}' contributing {impact_val}% attribution impact."
        elif risk_score > 0.3:
            return f"CAUTION: Component degradation observed. Key contributing factor: '{top_feature}' ({impact_val}% impact)."
        else:
            return "STABLE: All primary telemetry parameters operate well within safe flight thresholds."