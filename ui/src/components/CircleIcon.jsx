import React from 'react';
import SvgIcon from '@mui/material/SvgIcon';

const ColoredCircleIcon = (props) => {
  return (
    <SvgIcon {...props}>
      <circle cx="12" cy="12" r="10" />
    </SvgIcon>
  );
};

export default ColoredCircleIcon;