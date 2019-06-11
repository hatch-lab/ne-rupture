function FrameStats=get_frame_features(I,O,t)

  %This function is used to segment individual nuclei based on nlsGFP signal,
  % to define the cytoplasmic compartment, and to collect useful metrics for
  % each cell.
  % Input Arguments:
  %
  % I  Processed image frame
  % O  Original image frame
  % t  Time frame index
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

  %% Measure object properties from nucleus and cytoplasmic components

  RawStatsNuc=regionprops(Lnuc, O, 'all');
  RawStatsCyto=regionprops(Lcyto, O, 'all');

  ProcessedStatsNuc=regionprops(Lnuc, I, 'all');
  ProcessedStatsCyto=regionprops(Lcyto, I, 'all');

  %combine properties in one structure
  FrameStats = cell(numel(RawStatsNuc),26);
  for j=1:numel(RawStatsNuc)
    FrameStats(j,:) = {
      strcat(num2str(t),".",num2str(j)), % particle_id
      t, % frame

      RawStatsNuc(j).Centroid(1), % x
      RawStatsNuc(j).Centroid(2), % y

      RawStatsNuc(j).WeightedCentroid(1), % weighted_x_raw_nuc
      RawStatsNuc(j).WeightedCentroid(2), % weighted_y_raw_nuc
      RawStatsCyto(j).WeightedCentroid(1), % weighted_x_raw_cyto
      RawStatsCyto(j).WeightedCentroid(2), % weighted_y_raw_cyto
      ProcessedStatsNuc(j).WeightedCentroid(1), % weighted_x_proc_nuc
      ProcessedStatsNuc(j).WeightedCentroid(2), % weighted_y_proc_nuc
      ProcessedStatsCyto(j).WeightedCentroid(1), % weighted_x_proc_cyto
      ProcessedStatsCyto(j).WeightedCentroid(2), % weighted_y_proc_cyto

      RawStatsNuc(j).Area, % area_nuc
      RawStatsCyto(j).Area, % area_cyto

      RawStatsNuc(j).MeanIntensity, % mean_raw_nuc
      RawStatsCyto(j).MeanIntensity, % mean_raw_cyto
      ProcessedStatsNuc(j).MeanIntensity, % mean_proc_nuc
      ProcessedStatsCyto(j).MeanIntensity, % mean_prc_cyto

      RawStatsNuc(j).MinIntensity, % min_raw_nuc
      RawStatsCyto(j).MinIntensity, % min_raw_cyto
      ProcessedStatsNuc(j).MinIntensity, % min_proc_nuc
      ProcessedStatsCyto(j).MinIntensity, % min_prc_cyto 

      RawStatsNuc(j).MaxIntensity, % max_raw_nuc
      RawStatsCyto(j).MaxIntensity, % max_raw_cyto
      ProcessedStatsNuc(j).MaxIntensity, % max_proc_nuc
      ProcessedStatsCyto(j).MaxIntensity % max_prc_cyto      
    };
  end
