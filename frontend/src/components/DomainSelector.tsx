import React from 'react';
import { DomainsMap } from '../types';

interface DomainSelectorProps {
  domains: DomainsMap;
  selectedDomain: string;
  onSelectDomain: (domainId: string) => void;
}

export const DomainSelector: React.FC<DomainSelectorProps> = ({
  domains,
  selectedDomain,
  onSelectDomain,
}) => {
  return (
    <div style={{ marginBottom: '20px' }}>
      <label htmlFor="domain-select" style={{ fontWeight: 'bold', marginRight: '10px' }}>
        Choisir un domaine d'expertise :
      </label>
      <select
        id="domain-select"
        value={selectedDomain}
        onChange={(e) => onSelectDomain(e.target.value)}
        style={{ padding: '8px 12px', fontSize: '14px', borderRadius: '4px', border: '1px solid #ccc' }}
      >
        {Object.entries(domains).map(([id, name]) => (
          <option key={id} value={id}>
            {name}
          </option>
        ))}
      </select>
    </div>
  );
};
