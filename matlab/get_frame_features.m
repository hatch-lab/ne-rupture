function FrameStats=get_frame_features(I,t,um,M)

  %This function is used to segment individual nuclei based on nlsGFP signal,
  % to define the cytoplasmic compartment, and to collect useful metrics for
  % each cell.
  % Input Arguments:
  %
  % I  Processed image frame
  % t  Time frame index
  % um The number of microns/px
  % M  Where to save the masks for this frame
  %
  % Output argument:
  %
  % StatsNuc   Structure containing metrics for each individual cell (one
  % cell per row)
  % Area: size of the nuclear object (in pixels)
  % Centroid: x and y coordinates of the centroid of the nucleus
  % Circularity: index reflecting circularity of the object
  % Solidity: index reflecting solidity of the object
  % PixelIdxList: linear indices of pixels of the nuclear object
  % PixelValues: intensity values of each pixels in the nuclear object
  % MeanIntensity: mean pixel intensity of the nucleus
  % CytoValues: intensity values of each pixels in the cytoplasmic object
  % CytoIntensity: mean pixel intensity of the cytoplasmic object
  % CellID: arbitrary cell ID of the cell
  % TimeFrame: time frame index

  %% Nuclear segmentation

  Ib=I-5; % basic global background subtraction
  [~, NUC]=segment_frame(Ib,8,1,50,0); % marker-controlled watershed segmentation
  NUC=bwmorph(NUC,'thicken',2); % slight thickening of the nuckear mask
  CELL=bwmorph(NUC,'thicken',5); % 5pix thickening of the nuclear mask to define cell mask (nucleus + cytoplasm

  %% Define cytoplsmic region

  Lcell=bwlabel(CELL); %convert binary image to label matrix

  m=max(Lcell(:)); %determine total number of cells
  Lnuc=zeros(size(I));
  Lcyto=zeros(size(I));

  % Generate cytoplasmic mask by logical operation and link cytoplasm to
  % nucleus
  for i=1:m
    DW=Lcell==i & NUC;
    Lnuc(DW)=i;
    CW=Lcell==i & ~DW;
    Lcyto(CW)=i;
  end

  % Write out mask
  save(strcat(M,"/",int2str(t)), 'Lnuc')

  %% Measure object properties from nucleus and cytoplasmic components
  ProcessedStatsNuc=regionprops(Lnuc, I, 'all');
  ProcessedStatsCyto=regionprops(Lcyto, I, 'all');

  %combine properties in one structure
  FrameStats = cell(numel(ProcessedStatsNuc),18);
  for j=1:numel(ProcessedStatsNuc)
    FrameStats(j,:) = {
      strcat(num2str(t),".",num2str(j)), % particle_id
      t, % frame

      ProcessedStatsNuc(j).WeightedCentroid(1)*um, % x
      ProcessedStatsNuc(j).WeightedCentroid(2)*um, % y

      ProcessedStatsCyto(j).WeightedCentroid(1)*um, % weighted_x_cyto
      ProcessedStatsCyto(j).WeightedCentroid(2)*um, % weighted_y_cyto
      ProcessedStatsNuc(j).Centroid(1)*um, % x_nuc
      ProcessedStatsNuc(j).Centroid(2)*um, % y_nuc
      ProcessedStatsCyto(j).Centroid(1)*um, % x_cyto
      ProcessedStatsCyto(j).Centroid(2)*um, % y_cyto

      ProcessedStatsNuc(j).Area, % area_nuc
      ProcessedStatsNuc(j).Area, % area_cyto

      ProcessedStatsNuc(j).MeanIntensity, % mean_nuc
      ProcessedStatsCyto(j).MeanIntensity, % mean_cyto

      ProcessedStatsNuc(j).MinIntensity, % min_nuc
      ProcessedStatsCyto(j).MinIntensity, % min_cyto 

      ProcessedStatsNuc(j).MaxIntensity, % max_nuc
      ProcessedStatsCyto(j).MaxIntensity % max_cyto      
    };
  end
